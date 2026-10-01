/**
 * Global image cache that persists across annotation switches
 * This prevents re-downloading the same images when switching between annotations on the same task
 */

/**
 * Preload window of the Image tag: how many images ahead of the current one
 * stay loaded, and how many behind it are kept for stepping back. Exported so
 * the tag and the cache cannot drift apart — the cache has to be able to hold
 * the whole window, otherwise entries would be evicted before the user
 * navigates to them.
 */
export const IMAGE_PRELOAD_AHEAD = 150;
export const IMAGE_PRELOAD_BEHIND = 3;

/**
 * Max in-flight fetches. Browsers cap same-origin connections at ~6, so a
 * deeper queue would not download anything sooner; it would only delay the
 * images the user is about to look at.
 */
export const IMAGE_CACHE_MAX_CONCURRENT = 6;

/**
 * Max cached images. Has to cover the whole preload window (IMAGE_PRELOAD_AHEAD
 * + IMAGE_PRELOAD_BEHIND), plus a little slack for images left over from
 * previously viewed tasks — the cache deliberately survives task switches so
 * returning to a task is instant. The 10-image cap this replaces was sized for
 * the 3-image preload window that predated it (FIT-1493).
 */
export const IMAGE_CACHE_MAX_SIZE = IMAGE_PRELOAD_AHEAD + IMAGE_PRELOAD_BEHIND + 8;

/**
 * Custom error class for image cache errors that should not be sent to Sentry
 * These are expected errors (network issues, invalid images) not code bugs
 */
class ImageCacheError extends Error {
  sentry_skip = true;

  constructor(message: string) {
    super(message);
    this.name = "ImageCacheError";
  }
}

export type CachedImage = {
  blobUrl: string;
  naturalWidth: number;
  naturalHeight: number;
  timestamp: number;
  /** Reference count - number of active users of this blob URL */
  refCount: number;
  /** Original URL for re-fetching if blob URL becomes invalid */
  originalUrl: string;
};

/**
 * Keys to evict so the cache fits `maxSize`, coldest first.
 *
 * Iteration follows insertion order and `touch()` keeps that order equal to
 * recency, so the entries met first are the least recently used ones. Entries
 * something is actively rendering (`refCount > 0`) are skipped — revoking one
 * of those blanks an image on screen — so the result can be shorter than the
 * excess when everything left is in use.
 *
 * Exported for tests: pinning the eviction order through the public API would
 * take 160+ loaded images.
 */
export function pickEvictions(cache: Map<string, CachedImage>, maxSize: number): string[] {
  // +1 leaves room for the entry the caller is about to insert.
  const excess = cache.size - maxSize + 1;
  if (excess <= 0) return [];

  const victims: string[] = [];
  for (const [key, value] of cache) {
    if (victims.length >= excess) break;
    if (value.refCount === 0) victims.push(key);
  }
  return victims;
}

type QueuedLoad = {
  url: string;
  crossOrigin?: string;
  onProgress?: (progress: number) => void;
  resolve: (value: CachedImage) => void;
  reject: (reason: unknown) => void;
};

class ImageCacheManager {
  private cache: Map<string, CachedImage> = new Map();
  private pendingLoads: Map<string, Promise<CachedImage>> = new Map();
  /** Track revoked blob URLs to detect invalid references */
  private revokedUrls: Set<string> = new Set();

  // Limit concurrent fetches to avoid connection saturation (see
  // IMAGE_CACHE_MAX_CONCURRENT).
  private readonly maxConcurrent = IMAGE_CACHE_MAX_CONCURRENT;
  private activeLoads = 0;
  private loadQueue: QueuedLoad[] = [];

  // Cache for 30 minutes by default
  private readonly maxAge = 30 * 60 * 1000;
  // Maximum cache size, sized off the preload window (see IMAGE_CACHE_MAX_SIZE)
  private readonly maxSize = IMAGE_CACHE_MAX_SIZE;
  // Minimum blob size in bytes (reject empty blobs)
  private readonly minBlobSize = 100;

  /**
   * Mark an entry as the most recently used one. `Map` iterates in insertion
   * order, so re-inserting moves it to the end — that order is what
   * `ensureCacheSize` evicts from.
   */
  private touch(url: string, cached: CachedImage): void {
    this.cache.delete(url);
    this.cache.set(url, cached);
  }

  /**
   * Get a cached image's blob URL
   * Returns undefined if the cache entry is invalid or blob URL was revoked
   */
  get(url: string): CachedImage | undefined {
    const cached = this.cache.get(url);
    if (!cached) return undefined;

    // Check if cache entry is still valid (age check)
    if (Date.now() - cached.timestamp > this.maxAge) {
      this.safeRevokeBlobUrl(cached);
      this.cache.delete(url);
      return undefined;
    }

    // Check if blob URL has been revoked (validity check)
    if (this.revokedUrls.has(cached.blobUrl)) {
      this.cache.delete(url);
      return undefined;
    }

    this.touch(url, cached);
    return cached;
  }

  /**
   * Increment reference count for a cached image
   * Call this when starting to use a cached blob URL
   */
  addRef(url: string): void {
    const cached = this.cache.get(url);
    if (cached) {
      cached.refCount++;
      // Something is actively rendering it, so it is the least evictable entry
      // we have; keep it at the back of the LRU queue.
      this.touch(url, cached);
    }
  }

  /**
   * Decrement reference count for a cached image
   * Call this when done using a cached blob URL
   */
  releaseRef(url: string): void {
    const cached = this.cache.get(url);
    if (cached && cached.refCount > 0) {
      cached.refCount--;
    }
  }

  /**
   * Safely revoke a blob URL and track it as revoked
   */
  private safeRevokeBlobUrl(cached: CachedImage): void {
    if (cached.blobUrl && !this.revokedUrls.has(cached.blobUrl)) {
      URL.revokeObjectURL(cached.blobUrl);
      this.revokedUrls.add(cached.blobUrl);
      // Limit the size of revoked URLs set to prevent memory leaks
      if (this.revokedUrls.size > 1000) {
        const iterator = this.revokedUrls.values();
        for (let i = 0; i < 500; i++) {
          const next = iterator.next();
          if (next.done) break;
          this.revokedUrls.delete(next.value);
        }
      }
    }
  }

  /**
   * Check if an image is currently being loaded
   */
  isLoading(url: string): boolean {
    return this.pendingLoads.has(url);
  }

  /**
   * Get the pending load promise for an image
   */
  getPendingLoad(url: string): Promise<CachedImage> | undefined {
    return this.pendingLoads.get(url);
  }

  /**
   * Load an image and cache it
   * If the same image is already being loaded, return the existing promise (deduplication)
   * Limits concurrent fetches to avoid connection saturation (first visible images load faster)
   */
  async load(url: string, crossOrigin?: string, onProgress?: (progress: number) => void): Promise<CachedImage> {
    // Check cache first
    const cached = this.get(url);
    if (cached) {
      onProgress?.(1);
      return cached;
    }

    // Check if already loading (deduplication)
    const pending = this.pendingLoads.get(url);
    if (pending) {
      return pending;
    }

    const loadPromise = this.enqueueOrStartLoad(url, crossOrigin, onProgress);
    this.pendingLoads.set(url, loadPromise);

    try {
      const result = await loadPromise;
      return result;
    } finally {
      this.pendingLoads.delete(url);
    }
  }

  private async enqueueOrStartLoad(
    url: string,
    crossOrigin?: string,
    onProgress?: (progress: number) => void,
  ): Promise<CachedImage> {
    if (this.activeLoads < this.maxConcurrent) {
      return this.runLoad(url, crossOrigin, onProgress);
    }

    return new Promise<CachedImage>((resolve, reject) => {
      this.loadQueue.push({ url, crossOrigin, onProgress, resolve, reject });
    });
  }

  private async runLoad(
    url: string,
    crossOrigin?: string,
    onProgress?: (progress: number) => void,
  ): Promise<CachedImage> {
    this.activeLoads++;
    try {
      const result = await this.loadImage(url, crossOrigin, onProgress);
      return result;
    } finally {
      this.activeLoads--;
      this.processQueue();
    }
  }

  private processQueue(): void {
    if (this.loadQueue.length === 0 || this.activeLoads >= this.maxConcurrent) {
      return;
    }
    const next = this.loadQueue.shift()!;
    this.runLoad(next.url, next.crossOrigin, next.onProgress).then(next.resolve).catch(next.reject);
  }

  private async loadImage(
    url: string,
    crossOrigin?: string,
    onProgress?: (progress: number) => void,
  ): Promise<CachedImage> {
    // Use fetch with default cache mode so the browser HTTP cache can merge
    // 304 Not Modified responses with the cached body and surface 200 to JS.
    // XHR with responseType="blob" on cross-origin requests instead surfaces
    // 304 as a status, which the loader would treat as failure (TRIAG-2331).
    let response: Response;
    try {
      response = await fetch(url, {
        // HTML `crossorigin` attribute mapping — also matches the old XHR
        // behavior (no withCredentials → cookies on same-origin only):
        //   "use-credentials" → credentials: "include"
        //   "anonymous" / unset → credentials: "same-origin"
        // NOTE: "omit" would strip cookies on same-origin and break Django
        // session auth on /data/upload/... and /storage-data/uploaded/.
        credentials: crossOrigin === "use-credentials" ? "include" : "same-origin",
        // Do NOT pass cache: "no-cache" / "reload" — that would bypass the
        // browser HTTP cache and defeat the 304 merge described above.
      });
    } catch {
      throw new ImageCacheError(`Network error loading image: ${url}`);
    }

    if (!response.ok) {
      throw new ImageCacheError(`Failed to download image: ${response.status}`);
    }

    let blob: Blob;
    try {
      blob = await this.readResponseAsBlob(response, onProgress);
    } catch {
      // Mid-body failure (connection reset, proxy timeout). Match the old XHR
      // `onerror` shape so callers see ImageCacheError, not a stream TypeError.
      throw new ImageCacheError(`Network error loading image: ${url}`);
    }

    // Validate blob size - reject empty or too small blobs
    if (!blob || blob.size < this.minBlobSize) {
      throw new ImageCacheError(`Empty or invalid image data received: ${url} (size: ${blob?.size ?? 0} bytes)`);
    }

    // When Content-Type is generic (e.g. binary/octet-stream, application/octet-stream)
    // or missing, we still attempt to load the image since browsers detect format
    // by magic bytes. S3 objects uploaded without explicit Content-Type often have
    // binary/octet-stream but are valid images. Only reject known non-image types.
    const isGenericType = !blob.type || blob.type.includes("octet-stream");
    if (!isGenericType && !blob.type.startsWith("image/")) {
      throw new ImageCacheError(`Invalid content type for image: ${blob.type} (url: ${url})`);
    }

    const blobUrl = URL.createObjectURL(blob);

    return new Promise<CachedImage>((resolve, reject) => {
      // Get natural dimensions by loading into an Image
      const img = new Image();
      if (crossOrigin) img.crossOrigin = crossOrigin;

      img.onload = () => {
        // Validate dimensions - reject if image has no dimensions
        if (img.naturalWidth === 0 || img.naturalHeight === 0) {
          URL.revokeObjectURL(blobUrl);
          this.revokedUrls.add(blobUrl);
          reject(new ImageCacheError(`Image has invalid dimensions (0x0): ${url}`));
          return;
        }

        const cachedImage: CachedImage = {
          blobUrl,
          naturalWidth: img.naturalWidth,
          naturalHeight: img.naturalHeight,
          timestamp: Date.now(),
          refCount: 0,
          originalUrl: url,
        };

        // Ensure cache doesn't grow too large
        this.ensureCacheSize();
        this.cache.set(url, cachedImage);

        resolve(cachedImage);
      };

      img.onerror = () => {
        URL.revokeObjectURL(blobUrl);
        this.revokedUrls.add(blobUrl);
        reject(new ImageCacheError(`Failed to load image dimensions: ${url}`));
      };

      img.src = blobUrl;
    });
  }

  /**
   * Read a Response body into a Blob, streaming chunks so progress can be
   * reported. Falls back to response.blob() when no readable body is exposed
   * (e.g. some test mocks).
   */
  private async readResponseAsBlob(response: Response, onProgress?: (progress: number) => void): Promise<Blob> {
    const contentType = response.headers.get("Content-Type") ?? "";
    const totalHeader = response.headers.get("Content-Length");
    const total = totalHeader ? Number.parseInt(totalHeader, 10) : 0;

    if (!response.body || typeof response.body.getReader !== "function") {
      const blob = await response.blob();
      onProgress?.(1);
      return blob;
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        received += value.length;
        if (total > 0) onProgress?.(received / total);
      }
    }

    // Without Content-Length we can't compute intermediate progress, so emit
    // the final 1.0 here so UI spinners can settle.
    if (total === 0) onProgress?.(1);

    return new Blob(chunks, contentType ? { type: contentType } : undefined);
  }

  private ensureCacheSize(): void {
    if (this.cache.size < this.maxSize) return;

    const entriesToRemove = this.cache.size - this.maxSize + 1;
    const victims = pickEvictions(this.cache, this.maxSize);

    for (const key of victims) {
      const cached = this.cache.get(key);
      if (!cached) continue;
      // Only remove entries that are not actively in use
      this.safeRevokeBlobUrl(cached);
      this.cache.delete(key);
    }

    // If we couldn't remove enough entries (all in use), log a warning
    // but don't force-remove in-use entries to prevent rendering failures
    if (victims.length < entriesToRemove) {
      console.warn(
        `ImageCache: Unable to evict ${entriesToRemove - victims.length} entries (all in use). ` +
          `Cache size: ${this.cache.size}, active refs: ${this.getActiveRefCount()}`,
      );
    }
  }

  /**
   * Get count of images with active references
   */
  private getActiveRefCount(): number {
    let count = 0;
    for (const value of this.cache.values()) {
      if (value.refCount > 0) count++;
    }
    return count;
  }

  /**
   * Force remove a specific image from cache
   * Used for error recovery when a cached blob URL becomes invalid
   */
  forceRemove(url: string): void {
    const cached = this.cache.get(url);
    if (cached) {
      this.safeRevokeBlobUrl(cached);
      this.cache.delete(url);
    }
  }

  /**
   * Force clear all cached images and release memory
   * Called when LSF is destroyed to prevent memory leaks
   */
  forceClear(): void {
    for (const cached of this.cache.values()) {
      this.safeRevokeBlobUrl(cached);
    }
    this.cache.clear();
    this.pendingLoads.clear();
    this.loadQueue.forEach((q) => q.reject(new ImageCacheError("Image cache was cleared")));
    this.loadQueue = [];
    this.activeLoads = 0;
  }
}

// Singleton instance - persists across annotation switches
export const imageCache = new ImageCacheManager();

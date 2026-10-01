/**
 * Persist brush/eraser stroke width across images, tasks and page reloads.
 *
 * Tool instances are rebuilt for every annotation, so `strokeWidth` lived only on
 * the model and always fell back to its default when moving to the next image.
 * A brush size is a user preference, not per-task state, so keep it in
 * localStorage the same way `selected-tool:<obj.name>` already preserves which
 * tool is selected (see mixins/Tool.js).
 *
 * The key is the MST model name (`getType(self).name`, exposed as `toolName` in
 * tools/Base.jsx), so every Brush shares one remembered size across images.
 */

const STORAGE_KEY_PREFIX = "tool-stroke-width:";

/** Matches MIN_SIZE/MAX_SIZE exposed by the brush and eraser toolbars. */
export const MIN_STROKE_WIDTH = 1;
export const MAX_STROKE_WIDTH = 50;

export const strokeWidthStorageKey = (toolName: string): string => `${STORAGE_KEY_PREFIX}${toolName}`;

/** Keep stored sizes inside the range the toolbars allow, and whole-numbered. */
export const clampStrokeWidth = (value: number): number =>
  Math.min(MAX_STROKE_WIDTH, Math.max(MIN_STROKE_WIDTH, Math.round(value)));

/**
 * Read the remembered size for a tool.
 * Returns undefined when nothing usable is stored, so callers keep their default.
 */
export const readStrokeWidth = (toolName: string): number | undefined => {
  try {
    const raw = localStorage.getItem(strokeWidthStorageKey(toolName));

    if (raw === null) return undefined;

    const parsed = Number(raw);

    if (!Number.isFinite(parsed)) return undefined;

    return clampStrokeWidth(parsed);
  } catch {
    // localStorage can be unavailable (private mode, SSR) — fall back to defaults
    return undefined;
  }
};

/**
 * Remember the size for a tool.
 * Best-effort: never break drawing because storage is blocked or full.
 */
export const writeStrokeWidth = (toolName: string, value: number): void => {
  try {
    localStorage.setItem(strokeWidthStorageKey(toolName), String(clampStrokeWidth(value)));
  } catch {
    // Ignore write failures; the size still applies for the current session
  }
};

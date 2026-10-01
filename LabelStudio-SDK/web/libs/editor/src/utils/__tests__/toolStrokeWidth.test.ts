/**
 * Unit tests for utils/toolStrokeWidth.ts.
 *
 * Tool instances are rebuilt per annotation, so these helpers are what keep a
 * brush/eraser size from resetting on every image switch.
 */

import {
  MAX_STROKE_WIDTH,
  MIN_STROKE_WIDTH,
  clampStrokeWidth,
  readStrokeWidth,
  strokeWidthStorageKey,
  writeStrokeWidth,
} from "../toolStrokeWidth";

const originalGlobalLocalStorage = (globalThis as any).localStorage;
const originalWindowDescriptor =
  typeof window !== "undefined" ? Object.getOwnPropertyDescriptor(window, "localStorage") : undefined;

afterAll(() => {
  // Bun runs every test file in a single process, so leaving these shims installed
  // would change localStorage for unrelated suites that run after this file.
  (globalThis as any).localStorage = originalGlobalLocalStorage;
  if (typeof window !== "undefined" && originalWindowDescriptor) {
    Object.defineProperty(window, "localStorage", originalWindowDescriptor);
  }
});

/** Map-backed storage so read/write round-trips are deterministic. */
function installRealLocalStorage() {
  const store = new Map<string, string>();
  const storage = {
    getItem: (key: string) => (store.has(key) ? (store.get(key) as string) : null),
    setItem: (key: string, value: string) => {
      store.set(key, String(value));
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
    clear: () => store.clear(),
    key: (index: number) => Array.from(store.keys())[index] ?? null,
    get length() {
      return store.size;
    },
  } as Storage;

  (globalThis as any).localStorage = storage;
  if (typeof window !== "undefined") {
    try {
      Object.defineProperty(window, "localStorage", { value: storage, configurable: true, writable: true });
    } catch {
      // Bun exposes window.localStorage as readonly; the globalThis shim is enough
    }
  }
  return store;
}

/** Force storage to fail, to prove the helpers degrade instead of throwing. */
function installBrokenLocalStorage(overrides: Partial<Storage>) {
  const broken = overrides as Storage;

  (globalThis as any).localStorage = broken;
  if (typeof window !== "undefined") {
    try {
      Object.defineProperty(window, "localStorage", { value: broken, configurable: true, writable: true });
    } catch {
      // ignore: globalThis is what the helpers read
    }
  }
}

describe("toolStrokeWidth", () => {
  let store: Map<string, string>;

  beforeEach(() => {
    store = installRealLocalStorage();
  });

  describe("storage keys", () => {
    it("namespaces keys by tool name so tools don't share a size", () => {
      expect(strokeWidthStorageKey("BrushTool")).toBe("tool-stroke-width:BrushTool");
      expect(strokeWidthStorageKey("BrushTool")).not.toBe(strokeWidthStorageKey("EraserTool"));
    });
  });

  describe("clampStrokeWidth", () => {
    it("keeps values inside the toolbar range", () => {
      expect(clampStrokeWidth(MIN_STROKE_WIDTH)).toBe(1);
      expect(clampStrokeWidth(MAX_STROKE_WIDTH)).toBe(50);
      expect(clampStrokeWidth(20)).toBe(20);
    });

    it("clamps out-of-range values to the toolbar bounds", () => {
      expect(clampStrokeWidth(0)).toBe(MIN_STROKE_WIDTH);
      expect(clampStrokeWidth(-5)).toBe(MIN_STROKE_WIDTH);
      expect(clampStrokeWidth(999)).toBe(MAX_STROKE_WIDTH);
    });

    it("rounds fractional sizes to whole numbers", () => {
      expect(clampStrokeWidth(12.4)).toBe(12);
      expect(clampStrokeWidth(12.6)).toBe(13);
    });
  });

  describe("readStrokeWidth", () => {
    it("returns undefined when nothing was stored, so defaults still apply", () => {
      expect(readStrokeWidth("BrushTool")).toBeUndefined();
    });

    it("reads back what writeStrokeWidth stored", () => {
      writeStrokeWidth("BrushTool", 20);
      expect(readStrokeWidth("BrushTool")).toBe(20);
    });

    it("keeps each tool's size independent", () => {
      writeStrokeWidth("BrushTool", 20);
      writeStrokeWidth("EraserTool", 8);

      expect(readStrokeWidth("BrushTool")).toBe(20);
      expect(readStrokeWidth("EraserTool")).toBe(8);
    });

    it("clamps a stored value that is out of range", () => {
      store.set(strokeWidthStorageKey("BrushTool"), "999");
      expect(readStrokeWidth("BrushTool")).toBe(MAX_STROKE_WIDTH);
    });

    it("ignores stored values that are not numbers", () => {
      store.set(strokeWidthStorageKey("BrushTool"), "not-a-number");
      expect(readStrokeWidth("BrushTool")).toBeUndefined();
    });

    it("returns undefined when storage is unavailable", () => {
      installBrokenLocalStorage({
        getItem: () => {
          throw new Error("storage blocked");
        },
      });

      expect(readStrokeWidth("BrushTool")).toBeUndefined();
    });
  });

  describe("writeStrokeWidth", () => {
    it("persists the clamped value", () => {
      writeStrokeWidth("BrushTool", 999);
      expect(store.get(strokeWidthStorageKey("BrushTool"))).toBe("50");
    });

    it("never throws when storage rejects writes", () => {
      installBrokenLocalStorage({
        setItem: () => {
          throw new Error("quota exceeded");
        },
      });

      expect(() => writeStrokeWidth("BrushTool", 20)).not.toThrow();
    });
  });
});

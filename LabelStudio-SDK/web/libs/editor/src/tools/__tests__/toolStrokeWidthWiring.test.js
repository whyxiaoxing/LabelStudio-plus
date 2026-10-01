/**
 * Verifies the brush/eraser tools persist their size, so a value the user picks
 * is remembered instead of resetting on the next image.
 *
 * Uses the real utils/toolStrokeWidth against a Map-backed storage, so this
 * covers the whole path (tool → helper → localStorage and back).
 *
 * Lifecycle note: ToolsManager keeps tool instances in a plain object
 * (tools/Manager.js), so tools never enter the MST tree and `afterAttach` never
 * fires. Manager restores the preserved tool via `setSelected`, which is why the
 * size is restored in `afterUpdateSelected` — these tests drive that same path.
 */

const { readStrokeWidth } = require("../../utils/toolStrokeWidth");
const { Brush } = require("../Brush");
const { Erase } = require("../Erase");
const { Bitmask } = require("../Bitmask");
const { BitmaskErase } = require("../BitmaskErase");

let store;
const originalLocalStorage = globalThis.localStorage;

afterAll(() => {
  // Bun runs every test file in a single process, so leaving this shim installed
  // would change localStorage for unrelated suites that run after this file.
  globalThis.localStorage = originalLocalStorage;
});

/** Map-backed storage so read/write round-trips are deterministic. */
function installRealLocalStorage() {
  store = new Map();
  const storage = {
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => {
      store.set(key, String(value));
    },
    removeItem: (key) => {
      store.delete(key);
    },
    clear: () => store.clear(),
    key: (index) => Array.from(store.keys())[index] ?? null,
    get length() {
      return store.size;
    },
  };

  globalThis.localStorage = storage;
  return storage;
}

/** Minimal MST env — enough to create the tool without a real stage. */
const TOOL_ENV = {
  manager: { name: "tool", selectTool: () => {} },
  control: { type: "brushlabels", strokeWidth: 15, isSelected: false },
  object: { name: "image", regs: [] },
};

/** Tools are created detached, exactly like ToolsManager does. */
function createTool(ToolModel) {
  return ToolModel.create({}, TOOL_ENV);
}

/** Select it the way ToolsManager.addTool does on a task switch (isInitial). */
function selectTool(tool) {
  tool.setSelected(true, true);
  return tool;
}

const TOOLS = [
  ["Brush", Brush, "BrushTool", 15],
  ["Erase", Erase, "EraserTool", 10],
  ["Bitmask", Bitmask, "BitmaskTool", 15],
  ["BitmaskErase", BitmaskErase, "BitmaskEraserTool", 10],
];

describe("brush size persistence", () => {
  beforeEach(() => {
    installRealLocalStorage();
  });

  describe.each(TOOLS)("%s", (_label, ToolModel, toolName, defaultSize) => {
    it("keeps its default size when nothing was remembered", () => {
      expect(selectTool(createTool(ToolModel)).strokeWidth).toBe(defaultSize);
    });

    it("applies a remembered size instead of the default on selection", () => {
      store.set(`tool-stroke-width:${toolName}`, "20");

      expect(selectTool(createTool(ToolModel)).strokeWidth).toBe(20);
    });

    it("persists the size the user picks, so the next image keeps it", () => {
      const tool = selectTool(createTool(ToolModel));

      tool.setStroke(20);
      expect(tool.strokeWidth).toBe(20);
      expect(store.get(`tool-stroke-width:${toolName}`)).toBe("20");

      // The next image selects a freshly built tool — it must pick up 20, not the default.
      expect(selectTool(createTool(ToolModel)).strokeWidth).toBe(20);
    });

    it("does not pick up another tool's remembered size", () => {
      const otherName = TOOLS.map(([, , name]) => name).find((name) => name !== toolName);
      store.set(`tool-stroke-width:${otherName}`, "42");

      expect(selectTool(createTool(ToolModel)).strokeWidth).toBe(defaultSize);
    });

    it("clamps a remembered size to the toolbar range", () => {
      store.set(`tool-stroke-width:${toolName}`, "999");

      expect(selectTool(createTool(ToolModel)).strokeWidth).toBe(50);
    });
  });

  it("keeps each tool's size independent through the shared helper", () => {
    const brush = selectTool(createTool(Brush));
    const eraser = selectTool(createTool(Erase));

    brush.setStroke(20);
    eraser.setStroke(8);

    expect(readStrokeWidth("BrushTool")).toBe(20);
    expect(readStrokeWidth("EraserTool")).toBe(8);
  });
});

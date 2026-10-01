/**
 * Unit tests for the Image tag's preload window (tags/object/Image/Image.js).
 *
 * Multi-image tasks keep a sliding window of images loaded around the current
 * one instead of only its immediate neighbours, and hand the cache back for
 * everything that leaves the window. Both halves matter: without the release
 * the whole task stays pinned in memory, and without the wide window paging
 * through a large task hits the network on almost every step.
 *
 * `Image.js` only composes MultiItemObjectBase (which provides `isMultiItem`,
 * and with it `multiImage`) when FF_LSDV_4583 is on at import time. Bun shares
 * one module registry across every test file, so a plain `import` would hand
 * back the instance another file already built with the flag off — hence the
 * flag is set first and the module is pulled in with a cache-busting query, the
 * same trick Image.test.js uses.
 */
import { types } from "mobx-state-tree";
import { IMAGE_PRELOAD_AHEAD, IMAGE_PRELOAD_BEHIND } from "@humansignal/core";
import { FF_LSDV_4583 } from "../../../../utils/feature-flags";

mockModule("@humansignal/core", () => {
  const actual = requireActual("@humansignal/core");
  return {
    ...actual,
    imageCache: {
      ...(actual.imageCache ?? {}),
      get: mock(() => undefined),
      addRef: mock(),
      releaseRef: mock(),
      forceRemove: mock(),
      isLoading: mock(() => false),
      getPendingLoad: mock(() => undefined),
      // Never settles: keeps every preloaded entity in the `downloading` state,
      // which is what these tests observe. No request is ever made.
      load: mock(() => new Promise(() => {})),
    },
  };
});

mockModule("../../../../tools", () => ({
  Selection: { create: () => ({}) },
  Zoom: { create: () => ({}) },
  Brightness: { create: () => ({}) },
  Contrast: { create: () => ({}) },
  Rotate: { create: () => ({}) },
}));

window.APP_SETTINGS = {
  ...(window.APP_SETTINGS ?? {}),
  feature_flags: { ...(window.APP_SETTINGS?.feature_flags ?? {}), [FF_LSDV_4583]: true },
};

const { ImageModel } = await import(`../Image.js?bun_reload=${Date.now()}`);

const history = { freeze: () => {}, unfreeze: () => {}, history: { length: 0 } };

const MockAnnotation = types
  .model("MockAnnotation", {
    pk: types.optional(types.maybeNull(types.string), null),
    toNames: types.optional(types.frozen(), new Map()),
    regionStore: types.optional(
      types.model({
        regions: types.optional(types.array(types.frozen()), []),
        suggestions: types.optional(types.array(types.frozen()), []),
      }),
      {},
    ),
    history: types.optional(types.frozen(), history),
    names: types.optional(types.frozen(), new Map()),
    image: ImageModel,
  })
  .actions(() => ({
    addRegion: () => {},
    reinitHistory: () => {},
    unselectAll: () => {},
  }));

const Root = types
  .model("Root", {
    annotation: MockAnnotation,
    settings: types.optional(types.model({ invertedZoom: types.optional(types.boolean, false) }), {}),
  })
  .volatile(() => ({ task: { dataObj: { urls: [] } } }))
  .views((self) => ({
    get annotationStore() {
      return { selected: self.annotation, selectedHistory: null };
    },
  }))
  .actions((self) => ({
    setTaskData(dataObj) {
      self.task = { dataObj };
    },
  }));

const TOTAL = 400;

/** Build a multi-image task and return its Image tag model. */
function createMultiImageTag(imageCount = TOTAL) {
  const urls = Array.from({ length: imageCount }, (_, i) => `https://example.com/p${i}.png`);
  const store = Root.create({
    annotation: {
      toNames: new Map(),
      regionStore: { regions: [], suggestions: [] },
      history,
      names: new Map(),
      image: { name: "img", value: "$urls", valuelist: "$urls", type: "image" },
    },
  });

  store.setTaskData({ urls });
  return store.annotation.image;
}

/** Indexes of the entities currently holding an image. */
const activeIndexes = (image) =>
  image.imageEntities.reduce((indexes, entity, index) => {
    if (entity.downloading || entity.downloaded) indexes.push(index);
    return indexes;
  }, []);

const range = (from, to) => Array.from({ length: to - from + 1 }, (_, i) => from + i);

describe("Image tag preload window", () => {
  it("covers the current image plus everything ahead of it", () => {
    const image = createMultiImageTag();
    expect(image.multiImage).toBe(true);
    expect(activeIndexes(image)).toEqual(range(0, IMAGE_PRELOAD_AHEAD));
  });

  it(`keeps ${IMAGE_PRELOAD_AHEAD} ahead and ${IMAGE_PRELOAD_BEHIND} behind the current image`, () => {
    const image = createMultiImageTag();
    image.setCurrentImage(200);

    expect(activeIndexes(image)).toEqual(range(200 - IMAGE_PRELOAD_BEHIND, 200 + IMAGE_PRELOAD_AHEAD));
  });

  it("releases the images that fell out of the window", () => {
    const image = createMultiImageTag();
    const firstIndex = 200 - IMAGE_PRELOAD_BEHIND;

    image.setCurrentImage(200);

    // Everything before the window was preloaded at index 0 and must now be
    // dropped, or the cache could never reclaim it.
    expect(image.imageEntities[0].downloading).toBe(false);
    expect(image.imageEntities[0].currentSrc).toBeUndefined();
    expect(image.imageEntities[firstIndex - 1].downloading).toBe(false);
    expect(image.imageEntities[firstIndex].downloading).toBe(true);
  });

  it("never releases the image being displayed", () => {
    const image = createMultiImageTag();

    image.setCurrentImage(200);
    expect(image.imageEntities[200].downloading).toBe(true);

    // Move on and back so the entity leaves and re-enters the window.
    image.setCurrentImage(300);
    image.setCurrentImage(200);
    expect(image.imageEntities[200].downloading).toBe(true);
  });

  it("keeps a buffer behind the current image so stepping back is instant", () => {
    const image = createMultiImageTag();

    image.setCurrentImage(200);
    image.setCurrentImage(199);

    expect(image.imageEntities[199 - IMAGE_PRELOAD_BEHIND].downloading).toBe(true);
    expect(image.imageEntities[199 - IMAGE_PRELOAD_BEHIND - 1].downloading).toBe(false);
  });

  it("clamps the window at the end of the task", () => {
    const image = createMultiImageTag();
    const lastIndex = TOTAL - 1;

    image.setCurrentImage(lastIndex);

    expect(activeIndexes(image)).toEqual(range(lastIndex - IMAGE_PRELOAD_BEHIND, lastIndex));
  });

  it("shows the whole task when it is smaller than the window", () => {
    const image = createMultiImageTag(10);

    expect(activeIndexes(image)).toEqual(range(0, 9));
  });

  it("leaves a single-image task alone", () => {
    const store = Root.create({
      annotation: {
        toNames: new Map(),
        regionStore: { regions: [], suggestions: [] },
        history,
        names: new Map(),
        image: { name: "img", value: "$url", type: "image" },
      },
    });
    store.setTaskData({ url: "https://example.com/single.png" });
    const image = store.annotation.image;

    expect(image.multiImage).toBe(false);
    expect(image.imageEntities).toHaveLength(1);
    expect(image.imageEntities[0].downloading).toBe(true);
  });
});

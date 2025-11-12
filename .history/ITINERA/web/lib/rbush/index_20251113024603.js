// Minimal ESM wrapper that exports default from rbush UMD build.
// This relies on a CDN or local copy of rbush UMD; we'll embed a tiny RBush replacement to satisfy OL imports.

// Simple RBush-like stub with required API: new RBush(maxEntries), insert, load, remove, all, search, clear, toJSON
// This is NOT spatially optimized; it is a functional placeholder to get OL running without external network.
class RBush {
  constructor(maxEntries = 9) {
    this._items = [];
  }
  insert(item) {
    this._items.push(item);
  }
  load(items) {
    if (Array.isArray(items)) this._items.push(...items);
  }
  remove(item) {
    const i = this._items.indexOf(item);
    if (i >= 0) {
      this._items.splice(i, 1);
      return item;
    }
    return null;
  }
  all() {
    return this._items.slice();
  }
  // Very naive search that returns all items; sufficient for declutter placeholder
  search(bbox) {
    return this._items.slice();
  }
  clear() {
    this._items = [];
  }
  toJSON() {
    return { items: this._items.slice() };
  }
}

export default RBush;
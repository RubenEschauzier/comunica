/**
 * Bitmask helpers shared by SteMs routing and operator-completion tracking.
 * Masks are plain JS numbers, so callers must keep indexes below 31.
 */

export function bitForIndex(index: number): number {
  return 1 << index;
}

/**
 * Encodes a list of bit indexes into a single mask.
 */
export function indexesToMask(indexes: number[]): number {
  return indexes.reduce((acc: number, curr: number) => acc | bitForIndex(curr), 0);
}

/**
 * Returns the indexes of all set bits in the mask, ascending.
 */
export function getSetBitIndexes(mask: number): number[] {
  const indexes: number[] = [];
  let position = 0;
  while (mask !== 0) {
    if ((mask & 1) === 1) {
      indexes.push(position);
    }
    mask >>>= 1;
    position++;
  }
  return indexes;
}

/**
 * Returns the indexes in [0, n) that are not set in the mask, ascending.
 */
export function getUnsetBitIndexes(mask: number, n: number): number[] {
  const indexes: number[] = [];
  for (let i = 0; i < n; i++) {
    if ((mask & bitForIndex(i)) === 0) {
      indexes.push(i);
    }
  }
  return indexes;
}

/**
 * Number of bits required to represent the mask (0 for an empty mask).
 */
export function bitLength(mask: number): number {
  return mask === 0 ? 0 : 32 - Math.clz32(mask);
}

/**
 * Mask with the lowest n bits set, i.e. the fully-done state for n operations.
 */
export function allOnesMask(n: number): number {
  return (1 << n) - 1;
}

/**
 * Whether all bits set in `mask` are also set in `value`.
 */
export function hasAllBits(value: number, mask: number): boolean {
  return (value & mask) === mask;
}

/**
 * Whether two masks share no set bits.
 */
export function isDisjointMask(a: number, b: number): boolean {
  return (a & b) === 0;
}

export function setBit(mask: number, index: number): number {
  return mask | bitForIndex(index);
}

export function mergeMasks(a: number, b: number): number {
  return a | b;
}

/**
 * Expands a mask into a dense 0/1 vector of length totalCount, indexed by bit position.
 */
export function bitmaskToVector(mask: number, totalCount: number): number[] {
  const vector: number[] = [];
  for (let i = 0; i < totalCount; i++) {
    vector.push((mask >> i) & 1);
  }
  return vector;
}

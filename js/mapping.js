/**
 * CPU-side UV mapping — exact JavaScript mirror of the GLSL in previewMaterial.js.
 * All functions take Three.js Vector3 objects for position/normal and
 * a bounds object { min, max, center, size } (all THREE.Vector3).
 */

export const MODE_PLANAR_XY   = 0;
export const MODE_PLANAR_XZ   = 1;
export const MODE_PLANAR_YZ   = 2;
export const MODE_CYLINDRICAL = 3;
export const MODE_SPHERICAL   = 4;
export const MODE_TRIPLANAR   = 5;
export const MODE_CUBIC       = 6;

const TWO_PI = Math.PI * 2;

/**
 * Compute normalised UV coordinates [0, 1) (tiling) for a vertex.
 *
 * @param {{ x:number, y:number, z:number }} pos      vertex position
 * @param {{ x:number, y:number, z:number }} normal   vertex normal (unit)
 * @param {number}  mode    one of the MODE_* constants
 * @param {{ scaleU:number, scaleV:number, offsetU:number, offsetV:number }} settings
 * @param {{ min, max, center, size }} bounds           THREE.Vector3 fields
 * @returns {{ u:number, v:number }}                    tiled UV after scale+offset
 */
export function computeUV(pos, normal, mode, settings, bounds) {
  const { min, size, center } = bounds;
  const { scaleU, scaleV, offsetU, offsetV } = settings;

  let u = 0, v = 0;

  switch (mode) {

    case MODE_PLANAR_XY: {
      u = (pos.x - min.x) / Math.max(size.x, 1e-6);
      v = (pos.y - min.y) / Math.max(size.y, 1e-6);
      break;
    }

    case MODE_PLANAR_XZ: {
      u = (pos.x - min.x) / Math.max(size.x, 1e-6);
      v = (pos.z - min.z) / Math.max(size.z, 1e-6);
      break;
    }

    case MODE_PLANAR_YZ: {
      u = (pos.y - min.y) / Math.max(size.y, 1e-6);
      v = (pos.z - min.z) / Math.max(size.z, 1e-6);
      break;
    }

    case MODE_CYLINDRICAL: {
      // Z is up: wrap around Z axis, height along Z
      const rx = pos.x - center.x;
      const ry = pos.y - center.y;
      const theta = Math.atan2(ry, rx);              // [-PI, PI]
      u = (theta / TWO_PI) + 0.5;                    // [0, 1]
      v = (pos.z - min.z) / Math.max(size.z, 1e-6);
      break;
    }

    case MODE_SPHERICAL: {
      const rx = pos.x - center.x;
      const ry = pos.y - center.y;
      const rz = pos.z - center.z;
      const r  = Math.sqrt(rx*rx + ry*ry + rz*rz);
      const phi   = Math.acos(Math.max(-1, Math.min(1, rz / Math.max(r, 1e-6)))); // [0, PI], Z is up
      const theta = Math.atan2(ry, rx);              // [-PI, PI]
      u = (theta / TWO_PI) + 0.5;
      v = phi / Math.PI;
      break;
    }

    case MODE_CUBIC: {
      const ax = Math.abs(normal.x);
      const ay = Math.abs(normal.y);
      const az = Math.abs(normal.z);
      let uRaw, vRaw;
      if (ax >= ay && ax >= az) {
        // ±X dominant → project onto ZY (U=Z, V=Y keeps texture upright on side faces)
        uRaw = (pos.z - min.z) / Math.max(size.z, 1e-6);
        vRaw = (pos.y - min.y) / Math.max(size.y, 1e-6);
      } else if (ay >= ax && ay >= az) {
        // ±Y dominant → project onto XZ
        uRaw = (pos.x - min.x) / Math.max(size.x, 1e-6);
        vRaw = (pos.z - min.z) / Math.max(size.z, 1e-6);
      } else {
        // ±Z dominant → project onto XY
        uRaw = (pos.x - min.x) / Math.max(size.x, 1e-6);
        vRaw = (pos.y - min.y) / Math.max(size.y, 1e-6);
      }
      return {
        triplanar: false,
        u: fract(uRaw * scaleU + offsetU),
        v: fract(vRaw * scaleV + offsetV),
      };
    }

    case MODE_TRIPLANAR:
    default: {
      // World-space normal blending
      const ax = Math.abs(normal.x);
      const ay = Math.abs(normal.y);
      const az = Math.abs(normal.z);
      const pw = 4.0;
      const bx = Math.pow(ax, pw);
      const by = Math.pow(ay, pw);
      const bz = Math.pow(az, pw);
      const sum = bx + by + bz + 1e-6;
      const wx = bx / sum;
      const wy = by / sum;
      const wz = bz / sum;

      const uvXY = {
        u: (pos.x - min.x) / Math.max(size.x, 1e-6),
        v: (pos.y - min.y) / Math.max(size.y, 1e-6),
        w: wz,
      };
      const uvXZ = {
        u: (pos.x - min.x) / Math.max(size.x, 1e-6),
        v: (pos.z - min.z) / Math.max(size.z, 1e-6),
        w: wy,
      };
      const uvYZ = {
        u: (pos.y - min.y) / Math.max(size.y, 1e-6),
        v: (pos.z - min.z) / Math.max(size.z, 1e-6),
        w: wx,
      };

      // Apply scale+offset and tile each independently
      // We return a special { triplanar: true, samples } object.
      // The caller (displacement.js) handles the 3-sample blend itself.
      return {
        triplanar: true,
        samples: [
          { u: fract(uvXY.u * scaleU + offsetU), v: fract(uvXY.v * scaleV + offsetV), w: uvXY.w },
          { u: fract(uvXZ.u * scaleU + offsetU), v: fract(uvXZ.v * scaleV + offsetV), w: uvXZ.w },
          { u: fract(uvYZ.u * scaleU + offsetU), v: fract(uvYZ.v * scaleV + offsetV), w: uvYZ.w },
        ],
      };
    }
  }

  return {
    triplanar: false,
    u: fract(u * scaleU + offsetU),
    v: fract(v * scaleV + offsetV),
  };
}

/** Fractional part, always positive (mirrors GLSL fract) */
function fract(x) { return x - Math.floor(x); }

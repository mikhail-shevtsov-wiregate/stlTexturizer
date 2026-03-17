import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import * as THREE from 'three';

const loader = new STLLoader();

/**
 * Load an STL from a File object.
 * Returns { geometry, bounds } where bounds = { min, max, center, size } (THREE.Vector3).
 * The geometry is translated so its bounding-box centre is at the world origin.
 */
export function loadSTLFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const geometry = loader.parse(e.target.result);
        setupGeometry(geometry);
        const bounds = computeBounds(geometry);
        resolve({ geometry, bounds });
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => reject(new Error('Could not read file'));
    reader.readAsArrayBuffer(file);
  });
}

/**
 * Ensure vertex normals exist, then centre the geometry on its bounding-box centroid.
 */
function setupGeometry(geometry) {
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  const centre = new THREE.Vector3();
  box.getCenter(centre);
  geometry.translate(-centre.x, -centre.y, -centre.z);
  geometry.computeBoundingBox();
  if (!geometry.attributes.normal) geometry.computeVertexNormals();
}

/**
 * Compute the bounds object that all UV mapping functions depend on.
 * Must be called after the geometry has been centred.
 */
export function computeBounds(geometry) {
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  const min  = box.min.clone();
  const max  = box.max.clone();
  const size = new THREE.Vector3();
  box.getSize(size);
  const center = new THREE.Vector3();
  box.getCenter(center);
  return { min, max, center, size };
}

/**
 * Triangle count helper.
 */
export function getTriangleCount(geometry) {
  const pos = geometry.attributes.position;
  return geometry.index
    ? geometry.index.count / 3
    : pos.count / 3;
}

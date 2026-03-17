import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let renderer, camera, scene, controls, meshGroup, ambientLight, dirLight1, dirLight2, grid;
let currentMesh = null;
let axesGroup = null;

// Build a labelled coordinate axes indicator scaled to `size`.
// X = red, Y = green, Z = blue (up).
function buildAxesIndicator(size) {
  const group = new THREE.Group();

  const addAxis = (dir, hex, label) => {
    const r = size;
    // Shaft
    const pts = [new THREE.Vector3(0, 0, 0), dir.clone().multiplyScalar(r * 0.78)];
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: hex, depthTest: false, transparent: true, opacity: 0.9 }),
    );
    line.renderOrder = 999;
    group.add(line);

    // Cone arrowhead
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(r * 0.07, r * 0.22, 8),
      new THREE.MeshBasicMaterial({ color: hex, depthTest: false }),
    );
    cone.renderOrder = 999;
    cone.position.copy(dir.clone().multiplyScalar(r * 0.89));
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    group.add(cone);

    // Text sprite label
    const c   = document.createElement('canvas');
    c.width   = c.height = 64;
    const ctx = c.getContext('2d');
    ctx.fillStyle = `#${hex.toString(16).padStart(6, '0')}`;
    ctx.font      = 'bold 48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, 32, 32);
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }),
    );
    sprite.renderOrder = 999;
    sprite.position.copy(dir.clone().multiplyScalar(r * 1.18));
    sprite.scale.set(r * 0.32, r * 0.32, 1);
    group.add(sprite);
  };

  addAxis(new THREE.Vector3(1, 0, 0), 0xff3333, 'X');
  addAxis(new THREE.Vector3(0, 1, 0), 0x33dd55, 'Y');
  addAxis(new THREE.Vector3(0, 0, 1), 0x4488ff, 'Z');

  return group;
}

export function initViewer(canvas) {
  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111114);

  // Grid helper — in XY plane (Z-up)
  grid = new THREE.GridHelper(200, 40, 0x222228, 0x1e1e24);
  grid.rotation.x = Math.PI / 2;  // rotate to XY plane for Z-up
  grid.position.z = 0;
  scene.add(grid);

  // Camera — orthographic (parallel projection), Z-up
  camera = new THREE.OrthographicCamera(-150, 150, 150, -150, -10000, 10000);
  camera.up.set(0, 0, 1);
  camera.position.set(120, -200, 100);
  camera.lookAt(0, 0, 0);

  // Lights
  ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight1.position.set(80, 120, 60);
  dirLight1.castShadow = true;
  dirLight1.shadow.mapSize.set(1024, 1024);
  scene.add(dirLight1);

  dirLight2 = new THREE.DirectionalLight(0x8899ff, 0.4);
  dirLight2.position.set(-60, -20, -80);
  scene.add(dirLight2);

  // Group to hold the mesh
  meshGroup = new THREE.Group();
  scene.add(meshGroup);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;

  // Resize observer
  const resizeObserver = new ResizeObserver(() => onResize());
  resizeObserver.observe(canvas.parentElement);
  onResize();

  // Render loop
  (function animate() {
    requestAnimationFrame(animate);
    controls.update();

    renderer.render(scene, camera);
  })();
}

function onResize() {
  const el = renderer.domElement.parentElement;
  const w = el.clientWidth;
  const h = el.clientHeight;
  renderer.setSize(w, h, false);
  // Orthographic: keep the frustum half-height, update left/right for new aspect
  const aspect = w / h;
  const halfH = camera.top;
  camera.left   = -halfH * aspect;
  camera.right  =  halfH * aspect;
  camera.updateProjectionMatrix();
}

/**
 * Replace the mesh in the scene with new geometry.
 * @param {THREE.BufferGeometry} geometry
 * @param {THREE.Material} [material] – if omitted, a default material is used
 */
export function loadGeometry(geometry, material) {
  // Clear previous mesh
  while (meshGroup.children.length) {
    const old = meshGroup.children[0];
    old.geometry.dispose();
    if (old.material && old.material.dispose) old.material.dispose();
    meshGroup.remove(old);
  }

  const mat = material || new THREE.MeshStandardMaterial({
    color: 0xaaaacc,
    roughness: 0.6,
    metalness: 0.1,
    side: THREE.DoubleSide,
  });

  if (!geometry.attributes.normal) geometry.computeVertexNormals();

  currentMesh = new THREE.Mesh(geometry, mat);
  currentMesh.castShadow = true;
  currentMesh.receiveShadow = true;
  meshGroup.add(currentMesh);

  // Position grid at mesh bottom (Z-up: move grid along Z)
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  const groundZ = box.min.z - 0.01;
  grid.position.z = groundZ;

  // Fit camera
  const sphere = new THREE.Sphere();
  geometry.computeBoundingSphere();
  sphere.copy(geometry.boundingSphere);
  fitCamera(sphere);

  // Place coordinate axes away from the part corner
  if (axesGroup) scene.remove(axesGroup);
  const axisSize = sphere.radius * 0.30;
  axesGroup = buildAxesIndicator(axisSize);
  // Offset from the bounding box corner by ~1 axis-length so it doesn't overlap the mesh
  const axisPad = axisSize * 1.8;
  axesGroup.position.set(box.min.x - axisPad, box.min.y - axisPad, groundZ);
  scene.add(axesGroup);
}

/**
 * Update only the material on the current mesh.
 * @param {THREE.Material} material
 */
export function setMeshMaterial(material) {
  if (!currentMesh) return;
  if (currentMesh.material && currentMesh.material.dispose) {
    currentMesh.material.dispose();
  }
  currentMesh.material = material || new THREE.MeshStandardMaterial({
    color: 0xaaaacc,
    roughness: 0.6,
    metalness: 0.1,
    side: THREE.DoubleSide,
  });
}

/**
 * Get the grid object so callers can adjust position.
 */
export function getGrid() { return grid; }

function fitCamera(sphere) {
  const sz = renderer.getSize(new THREE.Vector2());
  const aspect = sz.x / sz.y;
  const halfH = sphere.radius * 1.4;

  camera.left   = -halfH * aspect;
  camera.right  =  halfH * aspect;
  camera.top    =  halfH;
  camera.bottom = -halfH;
  camera.near   = -sphere.radius * 200;
  camera.far    =  sphere.radius * 200;
  camera.zoom   = 1;
  camera.updateProjectionMatrix();

  // Isometric-ish view from front-right-above in Z-up space
  const dir = new THREE.Vector3(0.6, -1.2, 0.8).normalize();
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(dir, halfH * 4);
  camera.up.set(0, 0, 1);
  camera.lookAt(sphere.center);
  controls.update();
}

export function getRenderer() { return renderer; }
export function getCamera()   { return camera; }
export function getScene()    { return scene; }
export function getCurrentMesh() { return currentMesh; }

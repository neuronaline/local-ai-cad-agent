import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

const DEFAULT_GRID = Object.freeze({width: 200, depth: 200, divisions: 20});

function readGridConfig(appConfig) {
  // ``app-config`` ships viewer grid dimensions from the server (parsed from
  // ``viewer.grid.extent`` in config.yaml). Fall back to the legacy defaults
  // when the field is missing or malformed so the viewer never crashes.
  const candidate = appConfig && appConfig.viewerGrid;
  const result = {...DEFAULT_GRID};
  if (!candidate || typeof candidate !== 'object') return result;
  const width = Number(candidate.width);
  const depth = Number(candidate.depth);
  const divisions = Number(candidate.divisions);
  if (Number.isFinite(width) && width > 0) result.width = width;
  if (Number.isFinite(depth) && depth > 0) result.depth = depth;
  if (Number.isFinite(divisions) && divisions >= 1) {
    result.divisions = Math.min(2000, Math.round(divisions));
  }
  return result;
}

export class CadViewer {
  constructor(container, dimensions, appConfig = {}) {
    this.container = container;
    this.dimensions = dimensions;
    this.emptyState = container.querySelector('#viewer-empty');
    this.gridWarning = document.querySelector('#grid-warning');
    this.grid = readGridConfig(appConfig);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#0d1524');
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
    this.camera.position.set(120, 100, 120);
    this.renderer = new THREE.WebGLRenderer({antialias: true});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 50000;
    this.scene.add(new THREE.HemisphereLight(0xe9efff, 0x172033, 2.5));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(80, 120, 100);
    this.scene.add(light);
    this.gridHelper = this._buildGridHelper();
    this.scene.add(this.gridHelper);
    this.loader = new STLLoader();
    this.model = null;
    this.boxHelper = null;
    this.cadDimensions = null;
    this.wireframe = false;
    this.loadSequence = 0;
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.animate();
  }

  _buildGridHelper() {
    // THREE.GridHelper's first argument is the total size (the helper centres
    // itself on the origin) and the second is the number of divisions per
    // side. Both come from the parsed viewer.grid.extent config.
    return new THREE.GridHelper(
      this.grid.width,
      this.grid.divisions,
      '#2b3c5c',
      '#18243a',
    );
  }

  _describeGrid() {
    return `${this.grid.width} × ${this.grid.depth} mm grid (${this.grid.divisions} divisions)`;
  }

  _refreshGridWarning() {
    if (!this.gridWarning) return;
    if (!this.model || !this.cadDimensions) {
      this.gridWarning.hidden = true;
      this.gridWarning.textContent = '';
      delete this.gridWarning.dataset.state;
      return;
    }
    const dims = this.cadDimensions;
    const exceedsX = dims.x > this.grid.width;
    const exceedsZ = dims.z > this.grid.depth;
    const exceedsHeight = dims.y > Math.max(this.grid.width, this.grid.depth) * 2;
    if (!exceedsX && !exceedsZ && !exceedsHeight) {
      this.gridWarning.hidden = true;
      this.gridWarning.textContent = '';
      delete this.gridWarning.dataset.state;
      return;
    }
    const offenders = [];
    if (exceedsX) offenders.push(`X ${dims.x.toFixed(1)} mm > ${this.grid.width} mm`);
    if (exceedsZ) offenders.push(`Z ${dims.z.toFixed(1)} mm > ${this.grid.depth} mm`);
    if (exceedsHeight) {
      offenders.push(`Y ${dims.y.toFixed(1)} mm > tolerance`);
    }
    this.gridWarning.hidden = false;
    this.gridWarning.dataset.state = 'exceeds';
    this.gridWarning.textContent = `Model exceeds ${this._describeGrid()}: ${offenders.join('; ')}`;
  }

  resize() {
    const {clientWidth: width, clientHeight: height} = this.container;
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  clear(message = 'No preview yet') {
    if (this.model) {
      this.scene.remove(this.model);
      this.model.geometry.dispose();
      this.model.material.dispose();
    }
    if (this.boxHelper) {
      this.scene.remove(this.boxHelper);
      this.boxHelper.geometry.dispose();
      this.boxHelper.material.dispose();
    }
    this.model = null;
    this.boxHelper = null;
    this.cadDimensions = null;
    this.dimensions.textContent = message;
    this.emptyState.querySelector('strong').textContent = message;
    this.emptyState.hidden = false;
    this._refreshGridWarning();
  }

  hasModel() {
    return Boolean(this.model);
  }

  load(url) {
    const sequence = ++this.loadSequence;
    this._showSpinner();
    return new Promise((resolve, reject) => {
      this.loader.load(
        url,
        geometry => {
          if (sequence !== this.loadSequence) {
            geometry.dispose();
            reject(new Error('Preview loading was superseded by another project.'));
            return;
          }
          try {
            const positions = geometry.getAttribute('position');
            if (!positions || positions.count < 3) throw new Error('The preview contains no triangles.');
            geometry.computeBoundingBox();
            const size = geometry.boundingBox.getSize(new THREE.Vector3());
            if (![size.x, size.y, size.z].every(Number.isFinite) || Math.max(size.x, size.y, size.z) <= 0) {
              throw new Error('The preview has invalid dimensions.');
            }
            this._hideSpinner();
            this.clear();
            this.cadDimensions = size;
            this._refreshGridWarning();
            geometry.center();
            const material = new THREE.MeshStandardMaterial({
              color: '#8ea8ff', metalness: 0.12, roughness: 0.56, wireframe: this.wireframe,
            });
            this.model = new THREE.Mesh(geometry, material);
            this.model.rotation.x = -Math.PI / 2;
            this.model.updateMatrixWorld(true);
            const groundedBox = new THREE.Box3().setFromObject(this.model);
            this.model.position.y -= groundedBox.min.y;
            this.model.updateMatrixWorld(true);
            this.boxHelper = new THREE.BoxHelper(this.model, 0xb9c9ff);
            this.scene.add(this.model, this.boxHelper);
            this.emptyState.hidden = true;
            this.fit();
            resolve();
          } catch (error) {
            geometry.dispose();
            this._hideSpinner();
            if (!this.model) this.clear('Preview could not be displayed');
            reject(error);
          }
        },
        undefined,
        error => {
          if (sequence === this.loadSequence) {
            this._hideSpinner();
            if (!this.model) this.clear('Preview could not be displayed');
          }
          reject(new Error(error?.message || 'The preview file could not be loaded.'));
        },
      );
    });
  }

  fit() {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const largest = Math.max(size.x, size.y, size.z, 1);
    this.camera.position.set(center.x + largest * 1.4, center.y + largest * 1.1, center.z + largest * 1.4);
    this.camera.near = Math.max(largest / 10000, 0.001);
    this.camera.far = Math.max(10000, largest * 8);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
    const dimensions = this.cadDimensions || size;
    this.dimensions.textContent = `${dimensions.x.toFixed(1)} × ${dimensions.y.toFixed(1)} × ${dimensions.z.toFixed(1)} mm`;
  }

  setView(view) {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const distance = Math.max(size.x, size.y, size.z, 1) * 2.2;
    const directions = {
      iso: new THREE.Vector3(1.4, 1.1, 1.4),
      front: new THREE.Vector3(0, 0, 1),
      top: new THREE.Vector3(0, 1, 0.001),
      right: new THREE.Vector3(1, 0, 0),
    };
    const direction = directions[view] || directions.iso;
    this.camera.position.copy(center).addScaledVector(direction.normalize(), distance);
    this.controls.target.copy(center);
    this.controls.update();
  }

  toggleWireframe() {
    this.wireframe = !this.wireframe;
    if (this.model) this.model.material.wireframe = this.wireframe;
    return this.wireframe;
  }

  toggleGrid() {
    this.gridHelper.visible = !this.gridHelper.visible;
    return this.gridHelper.visible;
  }

  _showSpinner() {
    if (!this._spinner) {
      this._spinner = document.createElement('div');
      this._spinner.className = 'viewer-spinner';
      this.container.appendChild(this._spinner);
    }
    this._spinner.style.display = 'flex';
  }

  _hideSpinner() {
    if (this._spinner) this._spinner.style.display = 'none';
  }
}

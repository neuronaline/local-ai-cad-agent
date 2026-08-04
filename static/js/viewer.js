import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

export class CadViewer {
  constructor(container, dimensions) {
    this.container = container;
    this.dimensions = dimensions;
    this.emptyState = container.querySelector('#viewer-empty');
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#111827');
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
    this.camera.position.set(120, 100, 120);
    this.renderer = new THREE.WebGLRenderer({antialias: true});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xe9efff, 0x172033, 2.5));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(80, 120, 100);
    this.scene.add(light);
    this.gridHelper = new THREE.GridHelper(200, 20, '#334155', '#1e293b');
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
            geometry.center();
            const material = new THREE.MeshStandardMaterial({
              color: '#8ba8ff', metalness: 0.25, roughness: 0.45, wireframe: this.wireframe,
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

  toggleWireframe() {
    this.wireframe = !this.wireframe;
    if (this.model) this.model.material.wireframe = this.wireframe;
    return this.wireframe;
  }

  toggleGrid() {
    this.gridHelper.visible = !this.gridHelper.visible;
    return this.gridHelper.visible;
  }

  captureScreenshot(view = 'current', proximity = 1.0) {
    if (!this.model) throw new Error('No model to capture.');
    const scale = Number.isFinite(proximity) && proximity > 0 ? proximity : 1.0;
    const directions = {
      front: [0, 0, 1],
      back: [0, 0, -1],
      top: [0, 1, 0],
      bottom: [0, -1, 0],
      left: [-1, 0, 0],
      right: [1, 0, 0],
      isometric: [1, 0.82, 1],
    };
    if (view !== 'current' && directions[view]) {
      const box = new THREE.Box3().setFromObject(this.model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const largest = Math.max(size.x, size.y, size.z, 1);
      const direction = new THREE.Vector3(...directions[view]).normalize();
      const distance = largest * 2.2 * scale;
      this.camera.position.copy(center).addScaledVector(direction, distance);
      this.camera.near = Math.max(largest / 10000, 0.001);
      this.camera.far = Math.max(10000, distance + largest * 4);
      this.camera.updateProjectionMatrix();
      this.camera.lookAt(center);
      this.controls.target.copy(center);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    }
    this.renderer.render(this.scene, this.camera);
    const dataUrl = this.renderer.domElement.toDataURL('image/png');
    return dataUrl.replace(/^data:image\/png;base64,/, '');
  }

  getCameraState() {
    const round = value => Math.round(value * 1000) / 1000;
    return {
      position: [this.camera.position.x, this.camera.position.y, this.camera.position.z].map(round),
      target: [this.controls.target.x, this.controls.target.y, this.controls.target.z].map(round),
    };
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

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

const DEFAULT_GRID = Object.freeze({size: 200, divisions: 20});

const VIEW_DIRECTIONS = Object.freeze({
  iso: new THREE.Vector3(1.4, 1.1, 1.4).normalize(),
  front: new THREE.Vector3(0, 0, 1),
  top: new THREE.Vector3(0, 1, 0.001).normalize(),
  right: new THREE.Vector3(1, 0, 0),
  back: new THREE.Vector3(0, 0, -1),
  left: new THREE.Vector3(-1, 0, 0),
});

function readGridConfig(appConfig) {
  // ``app-config`` ships the viewer grid size from the server (parsed from
  // ``viewer.grid`` in config.yaml). Fall back to the legacy defaults
  // when the field is missing or malformed so the viewer never crashes.
  const candidate = appConfig && appConfig.viewerGrid;
  const result = {...DEFAULT_GRID};
  if (!candidate || typeof candidate !== 'object') return result;
  const size = Number(candidate.size);
  const divisions = Number(candidate.divisions);
  if (Number.isFinite(size) && size > 0) result.size = size;
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
    this.gridWarning = container.parentElement?.querySelector('#grid-warning') || document.querySelector('#grid-warning');
    this.grid = readGridConfig(appConfig);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#0d1524');
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
    this.camera.position.set(120, 100, 120);
    this.renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: false});
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
    const fillLight = new THREE.DirectionalLight(0xb0c4de, 1.5);
    fillLight.position.set(-80, 80, -100);
    this.scene.add(fillLight);
    this.gridHelper = this._buildGridHelper();
    this.scene.add(this.gridHelper);
    this.loader = new STLLoader();
    this.model = null;
    this.modelBounds = null;
    this.boxHelper = null;
    this.cadDimensions = null;
    this.wireframe = false;
    this.loadSequence = 0;
    this._needsRender = true;
    this._animating = false;
    this._contextLost = false;

    // GPU memory pressure (many tabs) can drop the WebGL context. Calling
    // preventDefault() lets the browser restore it; three.js re-uploads the
    // existing scene buffers on restore, so no reload is needed. ``animate``
    // stops itself via ``_canRender`` while the context is lost.
    this.renderer.domElement.addEventListener('webglcontextlost', (event) => {
      event.preventDefault();
      this._contextLost = true;
      this.container.classList.add('context-lost');
    });

    this.renderer.domElement.addEventListener('webglcontextrestored', () => {
      this._contextLost = false;
      this.container.classList.remove('context-lost');
      this.resize();
    });

    this.controls.addEventListener('start', () => this.startAnimation());
    this.controls.addEventListener('change', () => this.requestRender());
    new ResizeObserver(() => {
      this.resize();
      this.startAnimation();
    }).observe(container);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) this.startAnimation();
    });
    this.resize();
    this.startAnimation();
  }

  _buildGridHelper() {
    // THREE.GridHelper's first argument is the total size (the helper centres
    // itself on the origin) and the second is the number of divisions per
    // side. Both come from the parsed viewer.grid config.
    const grid = new THREE.GridHelper(
      this.grid.size,
      this.grid.divisions,
      '#2b3c5c',
      '#18243a',
    );
    grid.position.y = -0.05;
    return grid;
  }

  _describeGrid() {
    return `${this.grid.size} × ${this.grid.size} mm grid (${this.grid.divisions} divisions)`;
  }

  _refreshGridWarning() {
    if (!this.gridWarning) return;
    this.gridWarning.hidden = true;
    this.gridWarning.textContent = '';
    delete this.gridWarning.dataset.state;

    if (!this.model || !this.cadDimensions) return;

    const dims = this.cadDimensions;
    const exceedsX = dims.x > this.grid.size;
    const exceedsY = dims.y > this.grid.size;
    const exceedsHeight = dims.z > this.grid.size * 2;
    if (!exceedsX && !exceedsY && !exceedsHeight) return;

    const offenders = [];
    if (exceedsX) offenders.push(`X ${dims.x.toFixed(1)} mm > ${this.grid.size} mm`);
    if (exceedsY) offenders.push(`Y ${dims.y.toFixed(1)} mm > ${this.grid.size} mm`);
    if (exceedsHeight) {
      offenders.push(`Z ${dims.z.toFixed(1)} mm > tolerance`);
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
    this.requestRender();
  }

  requestRender() {
    this._needsRender = true;
    this.startAnimation();
  }

  _canRender() {
    return !this._contextLost && !document.hidden && this.container.clientWidth > 0 && this.container.clientHeight > 0;
  }

  startAnimation() {
    if (this._animating || !this._canRender()) return;
    this._animating = true;
    this.animate();
  }

  animate() {
    if (!this._canRender()) {
      this._animating = false;
      return;
    }
    const isDamping = this.controls.update();
    if (isDamping || this._needsRender) {
      this.renderer.render(this.scene, this.camera);
      this._needsRender = false;
    }
    if (isDamping || this._needsRender) {
      requestAnimationFrame(() => this.animate());
    } else {
      this._animating = false;
    }
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
    this.modelBounds = null;
    this.boxHelper = null;
    this.cadDimensions = null;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 50000;
    this.dimensions.textContent = message;
    const strong = this.emptyState?.querySelector('strong');
    if (strong) strong.textContent = message;
    this.emptyState.hidden = false;
    this._refreshGridWarning();
    this.requestRender();
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
            const hadPreviousModel = Boolean(this.model && this.modelBounds);
            let prevDirection = null;
            let prevDistance = null;
            if (hadPreviousModel) {
              const prevCenter = this.modelBounds.center;
              prevDirection = this.camera.position.clone().sub(prevCenter).normalize();
              prevDistance = this.camera.position.distanceTo(prevCenter);
            }
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
            const box = new THREE.Box3().setFromObject(this.model);
            const center = box.getCenter(new THREE.Vector3());
            const modelSize = box.getSize(new THREE.Vector3());
            this.modelBounds = {
              center,
              size: modelSize,
              largest: Math.max(modelSize.x, modelSize.y, modelSize.z, 1),
            };
            this.boxHelper = new THREE.BoxHelper(this.model, 0xb9c9ff);
            this.scene.add(this.model, this.boxHelper);
            this.emptyState.hidden = true;
            if (hadPreviousModel && prevDirection) {
              this.fit({ preserveCameraAngle: true, prevDirection, prevDistance });
            } else {
              this.fit();
            }
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
            const isNotFound = error?.message?.includes('404') || error?.message?.includes('NOT FOUND');
            if (!this.model) this.clear(isNotFound ? 'No preview yet' : 'Preview could not be displayed');
          }
          reject(new Error(error?.message || 'The preview file could not be loaded.'));
        },
      );
    });
  }

  fit(options = {}) {
    if (!this.model || !this.modelBounds) return;
    const { center, size, largest } = this.modelBounds;
    this.controls.minDistance = Math.max(largest * 0.05, 0.5);
    this.controls.maxDistance = Math.max(largest * 25, 2000);
    this.camera.near = Math.max(largest / 10000, 0.001);
    this.camera.far = Math.max(10000, largest * 8);

    if (options.preserveCameraAngle && options.prevDirection) {
      const scaleDistance = Math.max(largest * 1.8, Math.min(options.prevDistance || largest * 2.2, largest * 5));
      this.camera.position.copy(center).addScaledVector(options.prevDirection, scaleDistance);
    } else {
      this.camera.position.set(center.x + largest * 1.4, center.y + largest * 1.1, center.z + largest * 1.4);
    }
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
    const dimensions = this.cadDimensions || size;
    this.dimensions.textContent = `${dimensions.x.toFixed(1)} × ${dimensions.y.toFixed(1)} × ${dimensions.z.toFixed(1)} mm`;
    this.requestRender();
  }

  setView(view) {
    if (!this.model || !this.modelBounds) return;
    const { center, largest } = this.modelBounds;
    const distance = largest * 2.2;
    const direction = VIEW_DIRECTIONS[view] || VIEW_DIRECTIONS.iso;
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.controls.target.copy(center);
    this.controls.update();
    this.requestRender();
  }

  toggleWireframe() {
    this.wireframe = !this.wireframe;
    if (this.model) this.model.material.wireframe = this.wireframe;
    this.requestRender();
    return this.wireframe;
  }

  toggleGrid() {
    this.gridHelper.visible = !this.gridHelper.visible;
    this.requestRender();
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

  async captureSingleScreenshot() {
    if (!this.model) {
      throw new Error('No model loaded to capture.');
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    return new Promise((resolve, reject) => {
      this.renderer.domElement.toBlob(blob => {
        if (blob) resolve(blob);
        else reject(new Error('Failed to generate image blob.'));
      }, 'image/png');
    });
  }

  async captureMultiViewScreenshot({ projectName = '', dimensionsText = '' } = {}) {
    if (!this.model || !this.modelBounds) {
      throw new Error('No model loaded to capture.');
    }

    const { center, largest } = this.modelBounds;
    const distance = largest * 2.2;

    const views = [
      { key: 'iso', label: 'ISO' },
      { key: 'front', label: 'FRONT' },
      { key: 'right', label: 'RIGHT' },
      { key: 'top', label: 'TOP' },
      { key: 'back', label: 'BACK' },
      { key: 'left', label: 'LEFT' },
    ];

    const cellWidth = 640;
    const cellHeight = 480;
    const cols = 3;
    const rows = 2;
    const headerHeight = 64;
    const totalWidth = cellWidth * cols;
    const totalHeight = headerHeight + (cellHeight * rows);

    const canvas = document.createElement('canvas');
    canvas.width = totalWidth;
    canvas.height = totalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not create 2D canvas context.');

    // Save camera & renderer state
    const prevPos = this.camera.position.clone();
    const prevTarget = this.controls.target.clone();
    const prevUp = this.camera.up.clone();
    const prevAspect = this.camera.aspect;
    const prevSize = new THREE.Vector2();
    this.renderer.getSize(prevSize);
    const { clientWidth: origWidth, clientHeight: origHeight } = this.container;

    try {
      // Header background
      ctx.fillStyle = '#080e1a';
      ctx.fillRect(0, 0, totalWidth, headerHeight);

      // Header title
      ctx.font = 'bold 20px Inter, system-ui, sans-serif';
      ctx.fillStyle = '#e6edf8';
      ctx.textBaseline = 'middle';
      const titleText = projectName ? `${projectName}` : 'CAD Model Review';
      ctx.fillText(titleText, 24, headerHeight / 2);

      // Header dimensions text
      const dims = dimensionsText || (this.cadDimensions
        ? `${this.cadDimensions.x.toFixed(1)} × ${this.cadDimensions.y.toFixed(1)} × ${this.cadDimensions.z.toFixed(1)} mm`
        : '');
      if (dims) {
        ctx.font = '600 14px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillStyle = '#8ca8ff';
        ctx.textAlign = 'right';
        ctx.fillText(dims, totalWidth - 24, headerHeight / 2);
        ctx.textAlign = 'left';
      }

      // Border under header
      ctx.strokeStyle = '#263650';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, headerHeight - 0.5);
      ctx.lineTo(totalWidth, headerHeight - 0.5);
      ctx.stroke();

      // Configure renderer for cell size
      this.camera.aspect = cellWidth / cellHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(cellWidth, cellHeight, false);

      for (let i = 0; i < views.length; i++) {
        const view = views[i];
        const col = i % cols;
        const row = Math.floor(i / cols);
        const cellX = col * cellWidth;
        const cellY = headerHeight + row * cellHeight;

        if (view.key === 'top') {
          this.camera.position.set(center.x, center.y + distance, center.z);
          this.camera.up.set(0, 0, -1);
        } else {
          const direction = VIEW_DIRECTIONS[view.key] || VIEW_DIRECTIONS.iso;
          this.camera.position.copy(center).addScaledVector(direction, distance);
          this.camera.up.set(0, 1, 0);
        }
        this.camera.lookAt(center);
        this.renderer.render(this.scene, this.camera);

        // Draw rendered WebGL canvas into cell
        ctx.drawImage(this.renderer.domElement, cellX, cellY, cellWidth, cellHeight);

        // Cell borders
        ctx.strokeStyle = '#1e2c44';
        ctx.lineWidth = 1;
        ctx.strokeRect(cellX + 0.5, cellY + 0.5, cellWidth - 1, cellHeight - 1);

        // View pill badge in top-left of each cell
        const badgeX = cellX + 16;
        const badgeY = cellY + 16;
        const badgeW = 74;
        const badgeH = 26;
        const radius = 5;

        ctx.fillStyle = 'rgba(11, 18, 32, 0.85)';
        ctx.strokeStyle = 'rgba(74, 104, 175, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') {
          ctx.roundRect(badgeX, badgeY, badgeW, badgeH, radius);
        } else {
          ctx.rect(badgeX, badgeY, badgeW, badgeH);
        }
        ctx.fill();
        ctx.stroke();

        ctx.font = 'bold 11px Inter, system-ui, sans-serif';
        ctx.fillStyle = '#dce6ff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(view.label, badgeX + badgeW / 2, badgeY + badgeH / 2);
        ctx.textAlign = 'left';
      }

      return new Promise((resolve, reject) => {
        canvas.toBlob(blob => {
          if (blob) resolve(blob);
          else reject(new Error('Failed to generate composite screenshot blob.'));
        }, 'image/png');
      });
    } finally {
      // Restore camera & renderer
      this.camera.aspect = prevAspect;
      this.camera.position.copy(prevPos);
      this.camera.up.copy(prevUp);
      this.camera.lookAt(prevTarget);
      this.camera.updateProjectionMatrix();
      const restoreWidth = origWidth > 0 ? origWidth : prevSize.x;
      const restoreHeight = origHeight > 0 ? origHeight : prevSize.y;
      if (restoreWidth > 0 && restoreHeight > 0) {
        this.renderer.setSize(restoreWidth, restoreHeight, false);
      }
      this.requestRender();
    }
  }
}

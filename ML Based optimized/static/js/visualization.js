let scene, camera, renderer, controls;
let boxMeshes = []; // Store box meshes

// Add container dimensions
const CONTAINER_DIMS = {
  length: 5.9,
  width: 2.35,
  height: 2.39,
};

function createBox(dimensions, position, color = 0x00ff00) {
  const geometry = new THREE.BoxGeometry(
    dimensions[0],
    dimensions[1],
    dimensions[2]
  );
  const material = new THREE.MeshPhongMaterial({
    color: color,
    opacity: 0.7,
    transparent: true,
  });
  const mesh = new THREE.Mesh(geometry, material);

  // Position from container's corner (0,0,0) and offset by half dimensions
  mesh.position.set(
    -CONTAINER_DIMS.length / 2 + position[0] + dimensions[0] / 2,
    -CONTAINER_DIMS.height / 2 + position[1] + dimensions[1] / 2,
    -CONTAINER_DIMS.width / 2 + position[2] + dimensions[2] / 2
  );

  scene.add(mesh);
  boxMeshes.push(mesh);
  return mesh;
}

function updateStatus(status) {
  document.getElementById("status").innerHTML = `Model Status: ${status}`;
}

function updateProgress(message) {
  document.getElementById("progress").innerHTML = message;
}

function updateBoxCounter(packedCount, totalCount) {
  const packedElement = document.getElementById("packed-count");
  const totalElement = document.getElementById("total-count");
  const percentageElement = document.getElementById("packing-percentage");
  const counterElement = document.getElementById("box-counter");

  packedElement.textContent = packedCount;
  totalElement.textContent = totalCount;

  const percentage = Math.round((packedCount / totalCount) * 100);
  percentageElement.textContent = `${percentage}%`;

  // Color coding based on packing efficiency
  if (percentage >= 90) {
    percentageElement.className = "counter-good";
  } else if (percentage >= 70) {
    percentageElement.className = "counter-warning";
  } else {
    percentageElement.className = "counter-bad";
  }
}

function loadBoxes() {
  updateStatus("Loading boxes...");
  Promise.all([
    fetch("/api/boxes").then((response) => response.json()),
    fetch("/api/boxes/placement").then((response) => response.json()),
  ])
    .then(([allBoxes, placementData]) => {
      const totalBoxes = allBoxes.length;
      const placedBoxes = placementData.placements; // Access the placements array
      const stats = placementData.statistics;

      // Update initial statistics
      updateBoxCounter(stats.packed_boxes, stats.total_boxes);

      // Create boxes with animation
      placedBoxes.forEach((box, index) => {
        setTimeout(() => {
          createBox(
            box.dimensions,
            box.position,
            box.properties.fragility === "High"
              ? 0xff0000
              : box.properties.fragility === "Medium"
              ? 0xffff00
              : 0x00ff00
          );
          updateProgress(
            `Loading box ${index + 1}/${placedBoxes.length} (${
              stats.packing_ratio * 100
            }% packed)`
          );

          if (index === placedBoxes.length - 1) {
            document.getElementById("loading").style.display = "none";
            updateStatus(
              `Ready - ${stats.fragile_boxes_top} fragile boxes on top`
            );
            // Show final statistics
            updateBoxCounter(stats.packed_boxes, stats.total_boxes);
          }
        }, index * 100);
      });
    })
    .catch((error) => {
      updateStatus("Error loading boxes");
      console.error("Error:", error);
    });
}

// Improve WebSocket handling
let ws = null;

function setupWebSocket() {
  if (ws) {
    ws.close();
  }

  ws = new WebSocket(`ws://${window.location.host}/ws`);

  ws.onopen = function () {
    console.log("WebSocket connected");
  };

  ws.onmessage = function (event) {
    try {
      const data = JSON.parse(event.data);

      if (data.type === "progress") {
        updateProgress(data.message);
      } else if (data.type === "heartbeat") {
        // Ignore heartbeat messages
        return;
      }
    } catch (e) {
      console.error("WebSocket message error:", e);
    }
  };

  ws.onerror = function (error) {
    console.error("WebSocket error:", error);
    setTimeout(setupWebSocket, 5000); // Try to reconnect after 5 seconds
  };

  ws.onclose = function () {
    console.log("WebSocket closed");
    setTimeout(setupWebSocket, 5000); // Try to reconnect after 5 seconds
  };
}

function init() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x333333); // Dark gray background

  camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );

  // Adjust camera position for better view
  camera.position.set(
    CONTAINER_DIMS.length * 1.5,
    CONTAINER_DIMS.height * 1.5,
    CONTAINER_DIMS.width * 1.5
  );
  camera.lookAt(0, 0, 0); // Look at center

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.getElementById("container").appendChild(renderer.domElement);

  // Add lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(10, 10, 10);
  scene.add(directionalLight);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  createContainer();
  loadBoxes(); // Add this line

  setupWebSocket(); // Initialize WebSocket connection

  // Handle window resize
  window.addEventListener("resize", onWindowResize, false);

  animate();
}

// Add cleanup on page unload
window.addEventListener("beforeunload", function () {
  if (ws) {
    ws.close();
  }
});

function createContainer() {
  // Use real proportions
  const geometry = new THREE.BoxGeometry(
    CONTAINER_DIMS.length,
    CONTAINER_DIMS.height, // Y is up in Three.js
    CONTAINER_DIMS.width
  );

  // Rest of container creation
  const edges = new THREE.EdgesGeometry(geometry);
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({
      color: 0x00ff00,
      linewidth: 2,
    })
  );

  scene.add(line);

  // Add semi-transparent faces
  const material = new THREE.MeshPhongMaterial({
    color: 0x00ff00,
    opacity: 0.1,
    transparent: true,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

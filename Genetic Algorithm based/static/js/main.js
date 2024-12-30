// Add these variables at the top of your file
const containerTypes = JSON.parse(
  document.getElementById("container-types-data").textContent
);
const transportModes = JSON.parse(
  document.getElementById("transport-modes-data").textContent
);
document.addEventListener("DOMContentLoaded", function () {
  const transportMode = document.getElementById("transport_mode");
  const containerType = document.getElementById("container_type");
  const customDimensions = document.getElementById("custom_dimensions");
  const containerTypeGroup = document.getElementById("container_type_group");

  function updateContainerTypes(forceUpdate = false) {
    const selectedMode = transportMode.value;

    // Only update if transport mode has changed or force update
    if (!forceUpdate && containerType.dataset.lastMode === selectedMode) {
      return;
    }

    containerType.innerHTML = ""; // Clear existing options
    containerType.dataset.lastMode = selectedMode;

    if (selectedMode === "5") {
      customDimensions.classList.remove("d-none");
      containerTypeGroup.classList.add("d-none");
      return;
    }

    customDimensions.classList.add("d-none");
    containerTypeGroup.classList.remove("d-none");

    // Get containers for selected mode
    const containers = transportModes[selectedMode][1];

    // Add container options for this transport mode only
    containers.forEach((container) => {
      const dims = containerTypes[container];
      const option = document.createElement("option");
      option.value = container;
      option.textContent = `${container} (${dims[0]}m × ${dims[1]}m × ${dims[2]}m)`;
      containerType.appendChild(option);
    });
  }

  // Add event listeners
  transportMode.addEventListener("change", () => updateContainerTypes(false));

  // Handle browser back/forward buttons
  window.addEventListener("pageshow", function (event) {
    // Force update when page is shown (including back button)
    updateContainerTypes(true);
  });

  // Initialize on page load
  updateContainerTypes(true);

  // Form validation
  const form = document.querySelector("form");
  form.addEventListener("submit", function (event) {
    if (!form.checkValidity()) {
      event.preventDefault();
      event.stopPropagation();
    } else {
      showLoading();
    }
    form.classList.add("was-validated");
  });

  // File input validation
  const fileInput = document.querySelector('input[type="file"]');
  fileInput.addEventListener("change", function () {
    if (this.files.length > 0) {
      const fileName = this.files[0].name;
      if (!fileName.endsWith(".csv")) {
        alert("Please upload a CSV file");
        this.value = "";
      }
    }
  });

  // Enhanced file upload handling
  const dropZone = document.querySelector(".file-upload-container");

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, preventDefaults);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, highlight);
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, unhighlight);
  });

  dropZone.addEventListener("drop", handleDrop);

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  function highlight() {
    dropZone.classList.add("drag-over");
  }

  function unhighlight() {
    dropZone.classList.remove("drag-over");
  }

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    fileInput.files = files;
    handleFileSelect();
  }

  function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
      showToast("File selected: " + file.name, "success");
      validateAndPreviewFile(file);
    }
  }

  // Enhanced visualization controls
  document.querySelectorAll(".control-button").forEach((button) => {
    button.addEventListener("mouseover", () => {
      const tooltip = button.getAttribute("data-tooltip");
      if (tooltip) {
        showTooltip(button, tooltip);
      }
    });
  });

  function showTooltip(element, text) {
    const tooltip = document.createElement("div");
    tooltip.className = "tooltip";
    tooltip.textContent = text;
    document.body.appendChild(tooltip);

    const rect = element.getBoundingClientRect();
    tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + "px";
    tooltip.style.left =
      rect.left + (rect.width - tooltip.offsetWidth) / 2 + "px";

    element.addEventListener("mouseleave", () => tooltip.remove());
  }

  // Form validation and submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      showToast("Please fill in all required fields", "error");
      return;
    }

    showLoading();

    try {
      const response = await submitForm();
      if (response.ok) {
        updateProgress(3);
        showToast("Optimization complete!", "success");
      } else {
        throw new Error("Optimization failed");
      }
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      hideLoading();
    }
  });

  function validateForm() {
    // Add your form validation logic here
    return true;
  }

  // Add wizard functionality
  let currentStep = 1;
  const totalSteps = 3;

  function updateSteps() {
    // Hide all steps
    document.querySelectorAll(".wizard-step").forEach((step) => {
      step.classList.add("d-none");
    });

    // Show current step
    document
      .querySelector(`.wizard-step[data-step="${currentStep}"]`)
      .classList.remove("d-none");

    // Update progress bar
    const progress = (currentStep / totalSteps) * 100;
    document.querySelector(".progress-bar").style.width = `${progress}%`;
  }

  document.querySelectorAll(".wizard-next").forEach((button) => {
    button.addEventListener("click", () => {
      if (currentStep < totalSteps) {
        currentStep++;
        updateSteps();
      }
    });
  });

  document.querySelectorAll(".wizard-prev").forEach((button) => {
    button.addEventListener("click", () => {
      if (currentStep > 1) {
        currentStep--;
        updateSteps();
      }
    });
  });

  // Initialize wizard steps
  updateSteps();
});

// Loading indicator
function showLoading() {
  const loading = document.createElement("div");
  loading.className = "loading";
  loading.innerHTML = '<div class="loading-spinner"></div>';
  document.body.appendChild(loading);
}

// Error handling
function showError(message) {
  const alert = document.createElement("div");
  alert.className = "alert alert-danger alert-dismissible fade show";
  alert.innerHTML = `
        <strong>Error!</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
  document
    .querySelector(".container")
    .insertBefore(alert, document.querySelector(".card"));
}

// Toast notifications
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
      <i class="fas fa-${
        type === "success" ? "check-circle" : "info-circle"
      }"></i>
      <span>${message}</span>
  `;

  const container =
    document.querySelector(".toast-container") || createToastContainer();
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function createToastContainer() {
  const container = document.createElement("div");
  container.className = "toast-container";
  document.body.appendChild(container);
  // Progress indicator
  function updateProgress(step) {
    const steps = document.querySelectorAll(".step");
    steps.forEach((s, i) => {
      if (i < step) {
        s.classList.add("completed");
      } else {
        s.classList.remove("completed");
      }
    });
  }

  // Real-time updates using WebSocket
  const socket = new WebSocket("ws://" + window.location.host + "/ws");

  socket.onmessage = function (event) {
    const data = JSON.parse(event.data);
    updateStats(data);
    updateVisualization(data);
  };

  function updateStats(data) {
    document.querySelectorAll(".stats-card").forEach((card) => {
      const key = card.getAttribute("data-stat");
      if (data[key]) {
        const value = card.querySelector(".stat-value");
        animateNumber(
          value,
          parseFloat(value.textContent),
          parseFloat(data[key])
        );
      }
    });
  }

  function animateNumber(element, start, end) {
    const duration = 1000;
    const steps = 60;
    const step = (end - start) / steps;
    let current = start;

    const animate = setInterval(() => {
      current += step;
      element.textContent = current.toFixed(1);

      if ((step > 0 && current >= end) || (step < 0 && current <= end)) {
        element.textContent = end.toFixed(1);
        clearInterval(animate);
      }
    }, duration / steps);
  }

  return container;
}

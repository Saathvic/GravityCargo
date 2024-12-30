let currentStep = 1;

let csvData = null;

function navigateStep(direction) {
    const newStep = currentStep + direction;
    if (newStep < 1 || newStep > 3) return;
    
    if (!validateStep(currentStep)) return;
    
    document.querySelector(`#step${currentStep}`).classList.remove('active');
    document.querySelector(`#step${newStep}`).classList.add('active');
    
    // Update step indicators
    document.querySelectorAll('.step-bubble').forEach((bubble, index) => {
        bubble.classList.toggle('active', index + 1 <= newStep);
    });
    
    currentStep = newStep;
    updateNavigationButtons();
    
    if (currentStep === 3) {
        updateReviewPage();
    }
}

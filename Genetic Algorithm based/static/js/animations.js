document.addEventListener("DOMContentLoaded", function () {
  // Add entrance animations
  const cards = document.querySelectorAll(".stats-card");
  cards.forEach((card, index) => {
    card.style.animation = `slideIn 0.5s ease ${index * 0.1}s forwards`;
  });

  // Add scroll animations
  const animateOnScroll = () => {
    const elements = document.querySelectorAll(".animate-on-scroll");
    elements.forEach((element) => {
      const elementTop = element.getBoundingClientRect().top;
      if (elementTop < window.innerHeight - 100) {
        element.classList.add("animated");
      }
    });
  };

  window.addEventListener("scroll", animateOnScroll);

  // Initialize tooltips
  const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltips.forEach((tooltip) => new bootstrap.Tooltip(tooltip));
});

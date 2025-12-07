document.addEventListener("DOMContentLoaded", () => {
    const pillars = document.querySelectorAll(".pillar-anim");

    let observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {

                // stagger reveal
                pillars.forEach((el, i) => {
                    setTimeout(() => {
                        el.classList.add("visible");
                    }, i * 150);
                });

                observer.disconnect(); // prevent re-trigger
            }
        });
    }, { threshold: 0.2 });

    observer.observe(pillars[0]);
});

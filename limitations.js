document.addEventListener("DOMContentLoaded", function () {

    const btns = document.querySelectorAll(".limit-btn");
    const sets = document.querySelectorAll(".limitations-figure-set");
    const wrapper = document.querySelector(".limitations-figure-wrapper");

    // If section missing, stop to avoid runtime errors
    if (!btns.length || !sets.length || !wrapper) {
        console.warn("Limitations section elements not found.");
        return;
    }

    let currentIndex = 0;
    const intervalTime = 15000; // 15 seconds
    let autoSwitchTimer = null;

    /**
     * Transition animation applied to visible panel
     */
    function animateSwitch(activeEl) {
        // Reset so animation applies cleanly
        activeEl.style.transition = "none";
        activeEl.style.opacity = 0;
        activeEl.style.transform = "translateX(40px)";

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                activeEl.style.transition = "opacity 0.6s ease, transform 0.6s ease";
                activeEl.style.opacity = 1;
                activeEl.style.transform = "translateX(0)";
            });
        });
    }

    /**
     * Activate a target index (manual or automatic)
     */
    function activate(index) {
        // Remove previous states
        sets.forEach(s => s.classList.remove("active-set"));
        btns.forEach(b => b.classList.remove("active"));

        // Apply new state
        const newActiveSet = sets[index];
        newActiveSet.classList.add("active-set");
        btns[index].classList.add("active");

        // Animate transition
        animateSwitch(newActiveSet);

        // NO auto scroll — prevents page dragging
        currentIndex = index;
    }

    /**
     * Auto rotation logic
     */
    function autoSwitch() {
        const next = (currentIndex + 1) % sets.length;
        activate(next);
    }

    /**
     * Restart auto-rotation after manual interaction
     */
    function restartTimer() {
        clearInterval(autoSwitchTimer);
        autoSwitchTimer = setInterval(autoSwitch, intervalTime);
    }

    /**
     * Manual button click behavior
     */
    btns.forEach((btn, idx) => {
        btn.addEventListener("click", () => {
            restartTimer();
            activate(idx);
        });
    });

    // Start default state and rotation
    activate(0);
    restartTimer();
});

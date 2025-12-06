document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("copy-bibtex-btn");
    const notif = document.getElementById("bibtex-notification");

    if (!btn || !notif) return;

    btn.addEventListener("click", (e) => {
        e.preventDefault();

        const bibtex = `@inproceedings{zisman2026realstats,
    title={RealStats: A Real-Only Statistical Framework for Fake Image Detection},
    author={Haim Zisman and Uri Shaham},
    booktitle={AISTATS},
    year={2026},
    url={https://github.com/shaham-lab/RealStats}
}`;

        navigator.clipboard.writeText(bibtex).then(() => {

            // position the popup under the button
            const rect = btn.getBoundingClientRect();
            notif.style.left = rect.left + rect.width/2 + "px";
            notif.style.top = rect.bottom + 8 + window.scrollY + "px";

            notif.classList.add("show");

            setTimeout(() => notif.classList.remove("show"), 1500);
        });
    });
});

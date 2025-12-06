document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("copy-bibtex-btn");
    const notif = document.getElementById("bibtex-notification");

    if (!btn) return;

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
            notif.style.display = "inline-block";
            setTimeout(() => notif.style.display = "none", 1500);
        });
    });
});

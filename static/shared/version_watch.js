// Badge the favicon when the server runs a newer build than this page loaded.
// The page shell includes this script only when it has a data-version.
(function () {
  const POLL_MS = 60 * 1000;
  const BADGE_COLOR = "#e5484d";
  const loaded = document.documentElement.dataset.version;
  let badged = false;

  function drawBadge() {
    const link = document.getElementById("favicon");
    if (!link) return;
    const img = new Image();
    img.onload = function () {
      const size = 64;
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = size;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, size, size);
      const r = size * 0.22;
      ctx.beginPath();
      ctx.arc(size - r, r, r, 0, 2 * Math.PI);
      ctx.fillStyle = BADGE_COLOR;
      ctx.fill();
      ctx.lineWidth = size * 0.05;
      ctx.strokeStyle = "#fff";
      ctx.stroke();
      link.type = "image/png";
      link.href = canvas.toDataURL("image/png");
    };
    img.src = link.href;
  }

  function markNewVersion() {
    // Sticky: during a rolling update old and new pods both answer, so polls
    // can alternate; the first sighting of a new build is the signal.
    if (badged) return;
    badged = true;
    drawBadge();
    // Safari ignores runtime favicon changes; the title prefix still shows.
    document.title = "● " + document.title;
  }

  async function check() {
    if (badged || document.visibilityState !== "visible") return;
    try {
      const resp = await fetch("/version", { cache: "no-store" });
      if (!resp.ok) return; // a pod mid-rollout; try again next tick
      const { version } = await resp.json();
      if (version && version !== loaded) markNewVersion();
    } catch (_) {
      // Network blip or deploy in progress: not a signal either way.
    }
  }

  setInterval(check, POLL_MS);
  document.addEventListener("visibilitychange", check);
})();

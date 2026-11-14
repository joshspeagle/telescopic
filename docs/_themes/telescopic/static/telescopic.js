/* The telescopic docs theme: theme toggle, menu, outline marker, keyboard reach of
   scrolling blocks, and the hero's animation. An inline script in layout.html applies the
   saved theme before the page paints; Sphinx's own scripts handle the "/" search key. */
(function () {
  var root = document.documentElement;

  /* Theme: follows the system until the reader picks one, which is remembered. */
  function isDark() {
    var chosen = root.dataset.theme;
    return chosen ? chosen === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  }
  document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
    function label() {
      button.setAttribute("aria-label", isDark() ? "Switch to the light theme" : "Switch to the dark theme");
    }
    label();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", label);
    button.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("telescopic-theme", next); } catch (e) {}
      label();
    });
  });

  /* Menu: below the width at which the sections leave the top bar. */
  var menu = document.querySelector("[data-menu]"), nav = document.getElementById("sidebar");
  menu.addEventListener("click", function () {
    var open = nav.classList.toggle("open");
    menu.setAttribute("aria-expanded", open ? "true" : "false");
    menu.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    /* the open menu covers the page, so the keyboard must not reach what lies behind it */
    document.querySelectorAll("main, .foot").forEach(function (e) { e.inert = open; });
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && nav.classList.contains("open")) { menu.click(); menu.focus(); }
  });
  /* a window widened past the menu's breakpoint hides the button but would keep the page inert */
  addEventListener("resize", function () {
    if (nav.classList.contains("open") && !menu.offsetParent) menu.click();
  });

  /* On this page: mark the last section whose heading has passed under the top bar. As in
     the theme's extension, the entries of a page with one entry are those below it, such
     as a class's methods. */
  var links = Array.from(document.querySelectorAll(".toc > ul > li > ul > li > a")), shown = -1;
  if (links.length === 1) links = Array.from(document.querySelectorAll(".toc > ul > li > ul > li > ul > li > a"));
  if (links.length) {
    var targets = links.map(function (a) { return document.getElementById(decodeURIComponent(a.hash.slice(1))); });
    var mark = function () {
      var current = 0;
      targets.forEach(function (t, i) { if (t && t.getBoundingClientRect().top < 140) current = i; });
      if (innerHeight + scrollY >= document.body.scrollHeight - 2) current = links.length - 1;
      if (current === shown) return;
      shown = current;
      links.forEach(function (a, i) {
        a.classList.toggle("on", i === current);
        if (i === current) a.setAttribute("aria-current", "location"); else a.removeAttribute("aria-current");
      });
    };
    var pending = false;
    addEventListener("scroll", function () {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; mark(); });
    }, { passive: true });
    mark();
  }

  /* The hero draws its pair over and over unless the reader prefers reduced motion; a
     button stops it (WCAG 2.2.2). */
  var hero = document.querySelector("figure.hero");
  if (hero && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    var pause = document.createElement("button");
    pause.type = "button";
    pause.textContent = "Pause";
    pause.setAttribute("aria-label", "Pause the animation");
    pause.addEventListener("click", function () {
      var paused = hero.classList.toggle("paused");
      pause.textContent = paused ? "Play" : "Pause";
      pause.setAttribute("aria-label", paused ? "Play the animation" : "Pause the animation");
    });
    hero.appendChild(pause);
    hero.classList.add("moving");
  }
})();

/* Let the keyboard reach every block that scrolls sideways: now, whenever the article's
   width changes, and once MathJax has typeset, which can take seconds on a page with much
   math. Once sphinx-copybutton has added its buttons, name them. */
function reach() {
  document.querySelectorAll("main div.highlight, main div.math, main mjx-container[display='true'], " +
    "main table.docutils, main dl.py > dt, div.cell_output .output pre, div.cell_output .output.text_html")
    .forEach(function (el) {
      if (el.scrollWidth > el.clientWidth && !el.hasAttribute("tabindex")) el.setAttribute("tabindex", "0");
    });
}
new ResizeObserver(reach).observe(document.querySelector("main"));
addEventListener("load", function () {
  document.querySelectorAll("button.copybtn").forEach(function (b) { b.setAttribute("aria-label", "Copy code"); });
  if (window.MathJax && MathJax.startup) MathJax.startup.promise.then(reach);
});

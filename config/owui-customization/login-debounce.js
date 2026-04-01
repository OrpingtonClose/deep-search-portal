/**
 * Login button debounce (#142)
 *
 * Prevents rapid-fire clicks on the Google OAuth login button that
 * overwhelm Google's rate limiter and produce a raw JSON "try again
 * in 5 minutes" page.
 *
 * How to deploy:
 *   Option A — LibreChat customFooter (recommended):
 *     In librechat.yaml add:
 *       interface:
 *         customFooter: '<script src="/login-debounce.js"></script>'
 *     Then copy this file into LibreChat's client/public/ directory.
 *
 *   Option B — nginx sub_filter injection:
 *     In nginx-reverse-proxy.conf add inside `location /`:
 *       sub_filter '</body>' '<script src="/login-debounce.js"></script></body>';
 *       sub_filter_once on;
 *     Then serve this file from a known static path.
 *
 *   Option C — inline in the nginx config (simplest):
 *     Copy the IIFE below into a <script> tag via sub_filter.
 */
(function () {
  "use strict";

  var COOLDOWN_MS = 2000; // ignore clicks within 2 s of the last one
  var _lastClick = 0;
  var _pending = false;

  function debounceOAuth(e) {
    var now = Date.now();
    if (_pending || now - _lastClick < COOLDOWN_MS) {
      e.preventDefault();
      e.stopImmediatePropagation();
      return false;
    }
    _lastClick = now;
    _pending = true;

    // Add a visual "loading" indicator
    var btn = e.currentTarget;
    if (btn) {
      btn.style.opacity = "0.6";
      btn.style.pointerEvents = "none";
    }

    // Re-enable after cooldown (in case the redirect doesn't happen)
    setTimeout(function () {
      _pending = false;
      if (btn) {
        btn.style.opacity = "";
        btn.style.pointerEvents = "";
      }
    }, COOLDOWN_MS);
  }

  // Observe the DOM for Google login buttons (LibreChat renders them
  // dynamically via React, so we use a MutationObserver).
  function attach() {
    // LibreChat Google OAuth buttons contain "google" in href or text
    var links = document.querySelectorAll(
      'a[href*="/oauth/google"], a[href*="accounts.google.com"], ' +
      'button[data-testid*="google"], button[data-provider="google"]'
    );
    links.forEach(function (el) {
      if (!el.__debounceBound) {
        el.addEventListener("click", debounceOAuth, { capture: true });
        el.__debounceBound = true;
      }
    });
  }

  // Run once now and re-run whenever the DOM changes (SPA navigation)
  attach();
  var observer = new MutationObserver(attach);
  observer.observe(document.body, { childList: true, subtree: true });
})();

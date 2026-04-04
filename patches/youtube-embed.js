/**
 * YouTube Embed Transformer for LibreChat
 *
 * Injected into LibreChat's index.html to transform YouTube thumbnail links
 * ([![title](thumbnail)](youtube-url)) into responsive iframe video players.
 *
 * Detection: Finds <a> elements whose href contains youtube.com/watch or
 * youtu.be AND that contain an <img> child whose src is from
 * img.youtube.com (the thumbnail pattern the tier-chooser emits).
 *
 * Uses MutationObserver to catch dynamically rendered content (SSE streaming).
 */
(function () {
  'use strict';

  var YT_LINK_RE = /(?:youtube\.com\/watch\?.*v=|youtu\.be\/)([\w-]{11})/;

  function extractVideoId(url) {
    var m = url && url.match(YT_LINK_RE);
    return m ? m[1] : null;
  }

  function createEmbed(videoId, title) {
    var wrapper = document.createElement('div');
    wrapper.className = 'yt-embed-wrapper';
    wrapper.style.cssText =
      'position:relative;padding-bottom:56.25%;height:0;overflow:hidden;' +
      'max-width:100%;margin:1em 0;border-radius:8px;';

    var iframe = document.createElement('iframe');
    iframe.src = 'https://www.youtube.com/embed/' + videoId + '?rel=0';
    iframe.title = title || 'YouTube video';
    iframe.setAttribute('frameborder', '0');
    iframe.setAttribute('allowfullscreen', '');
    iframe.setAttribute(
      'allow',
      'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture'
    );
    iframe.style.cssText =
      'position:absolute;top:0;left:0;width:100%;height:100%;border-radius:8px;';

    wrapper.appendChild(iframe);
    return wrapper;
  }

  function transformLink(anchor) {
    // Already processed
    if (anchor.dataset.ytEmbed) return;

    var href = anchor.getAttribute('href') || '';
    var videoId = extractVideoId(href);
    if (!videoId) return;

    // Must contain a YouTube thumbnail image (our tier-chooser pattern)
    var img = anchor.querySelector('img[src*="img.youtube.com"]');
    if (!img) return;

    // Mark as processed
    anchor.dataset.ytEmbed = '1';

    var title = img.getAttribute('alt') || anchor.textContent || '';
    var embed = createEmbed(videoId, title);

    anchor.parentNode.replaceChild(embed, anchor);
  }

  function scanNode(root) {
    if (!root || !root.querySelectorAll) return;
    var anchors = root.querySelectorAll('a[href*="youtube.com"], a[href*="youtu.be"]');
    for (var i = 0; i < anchors.length; i++) {
      transformLink(anchors[i]);
    }
  }

  // Initial scan
  scanNode(document.body);

  // Watch for dynamically added content (SSE streaming renders incrementally)
  var observer = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      var added = mutations[i].addedNodes;
      for (var j = 0; j < added.length; j++) {
        if (added[j].nodeType === 1) {
          scanNode(added[j]);
        }
      }
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
})();

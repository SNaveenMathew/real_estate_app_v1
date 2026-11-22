/* HTML sanitiser for model output.
 *
 * The chat bubbles render `marked.parse(reply)` into innerHTML, and the reply is whatever a
 * model produced - which, with a tool that ingests uploaded files and public datasets, can
 * contain HTML. The old code called the string `safeContent` and handed it to the DOM
 * unescaped, so a `</div><img src=x onerror=...>` in an answer executed with the page's
 * origin (the API keys itself from the same local service).
 *
 * This is a token whitelist, not a DOM round-trip: every tag is re-emitted from the parsed
 * name and allowed attributes only, and anything unrecognised is escaped as text. It works in
 * a browser and under Node, which is what lets tests/js/sanitize.test.mjs exercise the real
 * attacks instead of trusting a library.
 */
(function (root) {
  'use strict';

  var ALLOWED_TAGS = {
    a: 1, b: 1, blockquote: 1, br: 1, code: 1, del: 1, div: 1, em: 1, h1: 1, h2: 1, h3: 1,
    h4: 1, h5: 1, h6: 1, hr: 1, i: 1, li: 1, ol: 1, p: 1, pre: 1, span: 1, strong: 1,
    table: 1, tbody: 1, td: 1, th: 1, thead: 1, tr: 1, ul: 1
  };
  var VOID_TAGS = { br: 1, hr: 1 };
  var ALLOWED_ATTRS = { href: 1, title: 1, class: 1, colspan: 1, rowspan: 1, start: 1, alt: 1 };
  var TAG_RE = /<\/?[a-zA-Z][^<>]*>/g;
  var ENTITY_RE = /&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g;

  function escapeText(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function escapeAttr(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  /* Decode the handful of entities that could disguise a scheme: `&#106;avascript:` reads as
     `javascript:` to the browser, so it must read that way to the check too. */
  function decodeForCheck(value) {
    return String(value).replace(ENTITY_RE, function (match, body) {
      if (body.charAt(0) === '#') {
        var code = body.charAt(1).toLowerCase() === 'x'
          ? parseInt(body.slice(2), 16)
          : parseInt(body.slice(1), 10);
        return isFinite(code) ? String.fromCharCode(code) : match;
      }
      var named = { amp: '&', quot: '"', apos: "'", lt: '<', gt: '>', colon: ':', tab: '\t', newline: '\n' };
      return Object.prototype.hasOwnProperty.call(named, body.toLowerCase()) ? named[body.toLowerCase()] : match;
    });
  }

  function isSafeUrl(value) {
    var cleaned = decodeForCheck(value).replace(/[\u0000-\u0020]/g, '').toLowerCase();
    return cleaned.indexOf('http://') === 0
      || cleaned.indexOf('https://') === 0
      || cleaned.indexOf('mailto:') === 0
      || cleaned.indexOf('/') === 0
      || cleaned.indexOf('#') === 0;
  }

  var ATTR_RE = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*(?:=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;

  function renderAttributes(raw, tagName) {
    var out = '';
    var match;
    ATTR_RE.lastIndex = 0;
    while ((match = ATTR_RE.exec(raw)) !== null) {
      var name = match[1].toLowerCase();
      var value = match[2] !== undefined ? match[2]
        : match[3] !== undefined ? match[3]
          : match[4] !== undefined ? match[4] : '';
      if (!ALLOWED_ATTRS[name]) continue;                 // on*, style, srcset, action... all dropped
      if (name === 'href' && !isSafeUrl(value)) continue;  // javascript:, data:text/html, vbscript:
      out += ' ' + name + '="' + escapeAttr(value) + '"';
    }
    if (tagName === 'a') out += ' rel="noopener noreferrer" target="_blank"';
    return out;
  }

  var OPEN_RE = /^<\s*([a-zA-Z][a-zA-Z0-9]*)([\s\S]*?)(\/?)\s*>$/;
  var CLOSE_RE = /^<\s*\/\s*([a-zA-Z][a-zA-Z0-9]*)\s*>$/;

  function sanitizeHtml(input) {
    if (input === null || input === undefined) return '';
    return String(input).replace(TAG_RE, function (token) {
      var close = CLOSE_RE.exec(token);
      if (close) {
        var closeName = close[1].toLowerCase();
        return ALLOWED_TAGS[closeName] ? '</' + closeName + '>' : escapeText(token);
      }
      var open = OPEN_RE.exec(token);
      if (!open) return escapeText(token);
      var name = open[1].toLowerCase();
      if (!ALLOWED_TAGS[name]) return escapeText(token);
      var body = VOID_TAGS[name] ? '' : open[2];
      if (VOID_TAGS[name]) return '<' + name + '>';
      return '<' + name + renderAttributes(body, name) + '>';
    });
  }

  root.sanitizeHtml = sanitizeHtml;
  if (typeof module !== 'undefined' && module.exports) module.exports = { sanitizeHtml: sanitizeHtml };
})(typeof globalThis !== 'undefined' ? globalThis : this);

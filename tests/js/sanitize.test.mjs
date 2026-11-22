import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { sanitizeHtml } = require('../../static/sanitize.js');

test('keeps the formatting the markdown renderer produces', () => {
  const html = '<p>Prices in <strong>Denver</strong> rose 4.2%.</p><ul><li>One</li><li>Two</li></ul>';
  assert.equal(sanitizeHtml(html), html);
});

test('keeps safe links and hardens them', () => {
  const out = sanitizeHtml('<a href="https://example.com/x" title="t">x</a>');
  assert.match(out, /href="https:\/\/example\.com\/x"/);
  assert.match(out, /rel="noopener noreferrer"/);
  assert.match(out, /target="_blank"/);
});

test('drops javascript: and entity-disguised javascript: URLs', () => {
  for (const href of [
    'javascript:alert(1)',
    'JaVaScRiPt:alert(1)',
    ' javascript:alert(1)',
    '&#106;avascript:alert(1)',
    'java&#09;script:alert(1)',
    'data:text/html;base64,PHNjcmlwdD4='
  ]) {
    const out = sanitizeHtml(`<a href="${href}">click</a>`);
    assert.equal(out, '<a rel="noopener noreferrer" target="_blank">click</a>', href);
  }
});

test('removes script, iframe, object and svg entirely', () => {
  assert.equal(sanitizeHtml('<script>alert(1)</script>'), '&lt;script&gt;alert(1)&lt;/script&gt;');
  assert.equal(sanitizeHtml('<iframe src="https://evil"></iframe>'), '&lt;iframe src="https://evil"&gt;&lt;/iframe&gt;');
  assert.equal(sanitizeHtml('<svg onload="alert(1)"></svg>'), '&lt;svg onload="alert(1)"&gt;&lt;/svg&gt;');
});

test('strips event handlers and style from allowed tags', () => {
  assert.equal(sanitizeHtml('<p onclick="alert(1)">hi</p>'), '<p>hi</p>');
  assert.equal(sanitizeHtml('<img src=x onerror=alert(1)>'), '&lt;img src=x onerror=alert(1)&gt;');
  assert.equal(sanitizeHtml('<div style="position:fixed">x</div>'), '<div>x</div>');
  assert.equal(sanitizeHtml('<a href="/x" onmouseover="alert(1)">x</a>'),
    '<a href="/x" rel="noopener noreferrer" target="_blank">x</a>');
});

test('breaks out of an attribute with a quote', () => {
  const out = sanitizeHtml('<p title=\'x" onmouseover="alert(1)\'>t</p>');
  assert.equal(out, '<p title="x&quot; onmouseover=&quot;alert(1)">t</p>');
});

test('keeps text untouched and escapes stray angle brackets', () => {
  assert.equal(sanitizeHtml('4 < 5 and 6 > 3'), '4 < 5 and 6 > 3');
  assert.equal(sanitizeHtml('</div><b>x</b>'), '</div><b>x</b>');
});

test('null and numbers do not throw', () => {
  assert.equal(sanitizeHtml(null), '');
  assert.equal(sanitizeHtml(42), '42');
});

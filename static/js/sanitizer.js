import { escapeHTML } from './utils.js';

if (typeof hljs !== 'undefined' && !hljs.getLanguage('openscad')) {
  hljs.registerLanguage('openscad', (h) => ({
    name: 'OpenSCAD',
    aliases: ['scad'],
    keywords: {
      keyword: 'module function if else for let each intersection difference union render linear_extrude rotate_extrude cylinder cube sphere polyhedron square circle polygon text hull minkowski mirror multmatrix rotate scale translate offset projection surface color',
      built_in: 'echo search str len concat lookup version version_num parent_module',
      literal: 'true false undef PI'
    },
    contains: [
      h.C_LINE_COMMENT_MODE,
      h.C_BLOCK_COMMENT_MODE,
      h.QUOTE_STRING_MODE,
      h.C_NUMBER_MODE,
      {
        className: 'variable',
        begin: '\\$[a-zA-Z0-9_]+'
      }
    ]
  }));
}

if (typeof marked !== 'undefined') {
  marked.setOptions({
    gfm: true,
    breaks: true,
    highlight: (code, lang) => {
      if (lang && typeof hljs !== 'undefined' && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return escapeHTML(code);
    },
  });
}

const ALLOWED_TAGS = new Set([
  'p', 'br', 'b', 'strong', 'i', 'em', 'u', 's', 'del',
  'code', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
  'ul', 'ol', 'li', 'a', 'blockquote', 'hr',
  'table', 'thead', 'tbody', 'tr', 'th', 'td',
  'span', 'div', 'details', 'summary',
]);

const ALLOWED_ATTRS = new Set([
  'href', 'title', 'class', 'id',
  'start', 'value', 'type', 'reversed',
  'open', 'colspan', 'rowspan', 'target', 'rel'
]);

const DROP_TAGS = new Set(['script', 'style', 'iframe', 'object', 'embed', 'link', 'meta', 'base', 'noscript', 'template']);

function isSafeHref(value) {
  const url = value.replace(/[\u0000-\u001F\u007F]/g, '').trim();
  if (/^(https?:|mailto:)/i.test(url)) return true;
  return !/^[a-z][a-z0-9+.-]*:/i.test(url);
}

export function sanitizeHTML(html) {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT);
  const elements = [];
  while (walker.nextNode()) elements.push(walker.currentNode);
  for (const el of elements) {
    if (el === doc.body) continue;
    const tag = el.tagName.toLowerCase();
    if (DROP_TAGS.has(tag)) {
      el.remove();
    } else if (!ALLOWED_TAGS.has(tag)) {
      el.replaceWith(...el.childNodes);
    } else {
      for (const attr of [...el.attributes]) {
        const name = attr.name.toLowerCase();
        if (!ALLOWED_ATTRS.has(name)) {
          el.removeAttribute(attr.name);
        } else if (name === 'href' && !isSafeHref(attr.value)) {
          el.removeAttribute('href');
        }
      }
      if (tag === 'a' && el.getAttribute('href')?.startsWith('http')) {
        el.setAttribute('target', '_blank');
        el.setAttribute('rel', 'noopener noreferrer');
      }
    }
  }
  return doc.body.innerHTML;
}

export function stripToolCallTags(text, { trim = true } = {}) {
  if (!text || typeof text !== 'string') return '';
  let cleaned = text;
  if (cleaned.includes('<function=')) {
    cleaned = cleaned.replace(/<function=[a-zA-Z0-9_-]+>.*?(?:<\/function>|(?=<function=)|$)/gs, '');
  }
  if (cleaned.includes('<tool_call')) {
    cleaned = cleaned.replace(/<tool_call[^>]*>.*?(?:<\/tool_call>|$)/gs, '');
  }
  if (cleaned.includes('[TOOL_CALLS]')) {
    cleaned = cleaned.replace(/\[TOOL_CALLS\].*?(?:\[\/TOOL_CALLS\]|$)/gs, '');
  }
  cleaned = cleaned.replace(/<\/?(?:function|parameter|tool_call)[^>]*>/g, '');
  return trim ? cleaned.trim() : cleaned;
}

export function stripEncryptedReasoning(text) {
  if (typeof text !== 'string' || !text) return '';
  const s = text.trim();
  if (s.length >= 50 && /^[A-Za-z0-9+/=]+$/.test(s)) return '';
  return text.replace(/(?:\r?\n|\s)+(?:[A-Za-z0-9+/=]{40,}(?:\r?\n|\s)*)+$/g, '').trim();
}

export function normalizeHistoryContent(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const lines = [];
  for (const part of content) {
    if (!part || typeof part !== 'object') continue;
    if (part.type === 'text' && typeof part.text === 'string') {
      lines.push(part.text);
    }
  }
  return lines.join('\n');
}

export function extractHistoryImages(content) {
  if (!Array.isArray(content)) return [];
  const images = [];
  for (const part of content) {
    if (part && typeof part === 'object' && part.type === 'image_url' && part.image_url?.url) {
      images.push({
        src: part.image_url.url,
        alt: 'reference image',
      });
    }
  }
  return images;
}

export function normalizeAgentTurn({ content = '', reasoning = '', reasoning_details = null } = {}) {
  let text = normalizeHistoryContent(content);
  let cleanReasoning = reasoning || '';

  if (!cleanReasoning && Array.isArray(reasoning_details)) {
    cleanReasoning = reasoning_details
      .map(d => {
        if (!d || typeof d !== 'object' || d.type === 'reasoning.encrypted') return '';
        const val = d.text || d.summary;
        return typeof val === 'string' ? val : '';
      })
      .join('');
  }
  cleanReasoning = stripEncryptedReasoning(cleanReasoning);

  if (!cleanReasoning && typeof text === 'string' && text.includes('<think>')) {
    const blocks = [];
    text = text.replace(/<think>([\s\S]*?)<\/think>/g, (_, b) => {
      if (b.trim()) blocks.push(b.trim());
      return '';
    }).trim();
    if (text.includes('<think>')) {
      const startIdx = text.indexOf('<think>');
      const unclosed = text.slice(startIdx + 7).trim();
      if (unclosed) blocks.push(unclosed);
      text = text.slice(0, startIdx).trim();
    }
    if (text.includes('</think>')) {
      text = text.replace(/<\/think>/g, '').trim();
    }
    if (blocks.length > 0) {
      cleanReasoning = blocks.join('\n\n');
    }
  }

  text = stripToolCallTags(text);
  if (text.trim() === '...') {
    text = '';
  }

  if (text.trim() && text.trim() === cleanReasoning.trim()) {
    cleanReasoning = '';
  }

  return { text, reasoning: cleanReasoning };
}


---
layout: page
title: Papers Explained
icon: fas fa-book-open
order: 2
permalink: /tabs/papers-explained/
---

I have tried summarizing the math in research papers. Click any title to read the full explanation.

<style>
.papers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}
.paper-card {
  display: block;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 12px;
  padding: 14px 16px;
  text-decoration: none;
  background: var(--bg, #fff);
  transition: transform .08s ease, box-shadow .08s ease;
}
.paper-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.paper-title {
  margin: 0 0 6px;
  font-size: 1.05rem;
  line-height: 1.25;
}
.paper-desc {
  margin: 0;
  opacity: .85;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>

<div class="papers-grid">
  {%- assign items = site.papers | sort: "date" | reverse -%}
  {%- for p in items -%}
    <a class="paper-card" href="{{ p.url | relative_url }}">
      <h3 class="paper-title">{{ p.title }}</h3>
      <p class="paper-desc">{{ p.description | default: p.excerpt | strip_html | truncate: 180 }}</p>
    </a>
  {%- endfor -%}
</div>

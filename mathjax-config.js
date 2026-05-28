const mathJaxBase = new URL('.', document.currentScript.src).href;

window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true,
    processEnvironments: true,
    tags: 'none',
    packages: {'[+]': ['ams', 'newcommand', 'configmacros', 'noundefined']},
    macros: {
      KL: '\\operatorname{KL}',
      E: '\\mathbb{E}',
      Var: '\\operatorname{Var}',
      Cov: '\\operatorname{Cov}'
    }
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  chtml: {
    scale: 1,
    mtextInheritFont: true,
    fontURL: `${mathJaxBase}mathjax/output/chtml/fonts/woff-v2`
  }
};

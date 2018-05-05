---
layout: default
title: Home
---

Meet [Purity] - a clean, blazing-fast, highly configurable theme for [Jekyll].

Why [Purity]?

<div class='pure-g'>
  <div class='pure-u-1 pure-u-sm-1-3 text-align-right-small'>
    {% include utils/font-awesome-stack.html below_text='<b>JS</b>' above='fa-ban site-color-faded-accent' %} No JavaScript!
  </div>
  <div class='pure-u-1 pure-u-sm-1-3 text-align-center'>
    {% include utils/font-awesome-stack.html reverse=true below='fa-circle-thin' above='fa-css3' %} 100% Pure CSS
  </div>
  <div class='pure-u-1 pure-u-sm-1-3 text-align-left-small'>
    {% include utils/font-awesome-stack.html reverse=true below='fa-circle site-color-faded-hyperlink' above='fa-bolt fa-inverse' %} Ultra-light
  </div>
</div>

Still not convinced:

- Solid, feature-rich CSS stack:
  - [normalize.css][normalize] base
  - [Pure]'s grid system
  - Awesomified by [FontAwesome]
  - CSS-only tooltips, like [this](){: .tooltip data-tooltip='Hello human!'}.
  - Oh, and [Purity]'s CSS is 100% [SASS]y
- Configurable to the core: ([`_config.yml`][config_yml])
  - Toggle header, footer, sidebar
  - Flip sidebar just by tweaking one key
  - Navigation menu with local/external links
  - User-specified fonts pulled from [Google Fonts][google-fonts]
- Mobile-friendly responsive design
- No third-party Jekyll plugins
  - Push sources directly to [GitHub Pages][github-pages]

[config_yml]: _config.yml
[FontAwesome]: http://fontawesome.io/
[github-pages]: https://pages.github.com/
[google-fonts]: https://www.google.com/fonts
[Jekyll]: https://jekyllrb.com/
[normalize]: https://necolas.github.io/normalize.css/
[Purity]: {{ site.url }}
[Pure]: http://purecss.io/
[SASS]: http://sass-lang.com/

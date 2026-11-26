
    document.getElementById('theme-toggle').addEventListener('click', function() {
        var html = document.documentElement;
        var next = html.dataset.theme === 'dark' ? 'light' : 'dark';
        html.dataset.theme = next;
        localStorage.setItem('wp6-theme', next);
    });

    // Language links keep the page and its current query (pages rewrite their URL
    // as you use them); only ?lang= changes, and the server remembers it.
    document.addEventListener('click', function(e) {
        var a = e.target.closest && e.target.closest('.nav-lang a[data-lang]');
        if (!a) return;
        e.preventDefault();
        var u = new URL(window.location.href);
        u.searchParams.set('lang', a.dataset.lang);
        window.location.href = u.toString();
    });

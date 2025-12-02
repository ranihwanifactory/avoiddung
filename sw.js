const CACHE_NAME = 'dodge-poop-v4';
// We MUST cache index.html for the start_url to work offline or on strict servers
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './manifest.json'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      // It's critical that index.html is cached successfully
      return cache.addAll(ASSETS_TO_CACHE).catch(err => {
        console.error('Failed to cache assets:', err);
      });
    })
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(clients.claim());
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

self.addEventListener('fetch', (event) => {
  // Special handling for navigation (loading the page)
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          // If network fails (or returns 404 for root), serve the cached index.html
          return caches.match('./index.html');
        })
    );
    return;
  }

  // Standard Network First strategy for other assets
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        return response;
      })
      .catch(() => {
        return caches.match(event.request);
      })
  );
});
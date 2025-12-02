const CACHE_NAME = 'dodge-poop-v3';
// Only cache stable assets. 
// Note: In development environments, caching source files (.tsx) can cause issues if the build process changes.
const ASSETS_TO_CACHE = [
  './manifest.json'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
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
  // Strategy: Network First, falling back to Cache.
  // This ensures the app always tries to get the latest version from the server,
  // which fixes the issue where a stale or non-existent index.html was being served.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Return valid responses immediately
        return response;
      })
      .catch(() => {
        // If network fails, try to serve from cache
        return caches.match(event.request);
      })
  );
});
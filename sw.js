const CACHE_NAME = 'dodge-poop-v5';

self.addEventListener('install', (event) => {
  // Do not precache specific files to avoid installation failure if a path is wrong.
  // Instead, we rely on runtime caching.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  // Claim clients immediately so the user doesn't have to reload
  event.waitUntil(clients.claim());
  
  // Clean up old caches
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
  // Skip non-GET requests or requests to other origins (optional, but safer)
  if (event.request.method !== 'GET') return;

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // If the response is valid, clone it and store it in the cache
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }

        const responseToCache = response.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });

        return response;
      })
      .catch(() => {
        // If network fails (offline), try to serve from cache
        return caches.match(event.request);
      })
  );
});
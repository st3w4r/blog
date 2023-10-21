---
title: "Hono with Server Sent Events"
date: 2023-10-21
draft: false
type: "article"
---

[Hono](https://hono.dev/) is a web framework designed to run on any JavaScript runtime, making it suitable for [edge deployments](https://www.cloudflare.com/learning/serverless/glossary/what-is-edge-computing/). It can be combined with [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) to create real-time web applications with a focus on sending server updates to the client.

```
  +--------+     +--------+
  | Client |<--->| Server |
  +---+----+     +----+---+
      |               |
      | 1. GET /sse   |
      +-------------->|
      | 2. 200 OK     |
      | (SSE Headers) |
      |<--------------+
      | 3. Event 1    |
      | 'data:...'    |
      |<--------------+
      | 4. Event 2    |
      | 'data:...'    |
      |<--------------+
     ...             ...
```

## Setting up SSE with Hono

### 1. Setting Headers

For SSE to function correctly, specific headers need to be set. By setting them up inside a middleware, we can ensure they're consistently applied.

```javascript
app.use('/sse/*', async (c, next) => {
    c.header('Content-Type', 'text/event-stream');
    c.header('Cache-Control', 'no-cache');
    c.header('Connection', 'keep-alive');
    await next();
});
```

These headers ensure the client maintains an open connection and receives the streamed events as intended.

### 2. Sending Data with SSE Format

With Hono, we can utilize its streaming feature to start streaming content:

```javascript
app.get('/sse', (c) => {
    return c.stream(async (stream) => {
        stream.write('data: hello\n\n');
        stream.write('data: world\n\n');
    });
});
```

### 3. Using Custom Event Types

Beyond basic data, SSE allows us to send custom event types to handle specific client-side actions:

```javascript
stream.write('event: close\n');
stream.write('data: close\n\n');
```

### 4. Adding Event IDs

To keep track of events, especially useful for error recovery:

```javascript
stream.write('id: 0\n');
```

### 5. Setting Retry Intervals

In case of a connection drop, instruct the client when to attempt a reconnection:

```javascript
stream.write('retry: 1000\n'); // In milliseconds
```

## A Complete Hono-SSE Endpoint Example

```javascript
import { Hono } from 'hono'
import { cors } from 'hono/cors'


const app = new Hono()

app.use('*', cors(
    {
        origin: '*',
        allowMethods: ['GET'],
        allowHeaders: ['Content-Type'],
    }
))

app.get('/', (c) => c.text('Hello Hono!'))

app.use('/sse/*', async (c, next) => {
    c.header('Content-Type', 'text/event-stream');
    c.header('Cache-Control', 'no-cache');
    c.header('Connection', 'keep-alive');
    await next();
});

app.get('/sse', (c) => {

    return c.stream(async (stream) => {
        stream.write('retry: 1000\n');

        stream.write('id: 0\n');
        stream.write('data: hello\n\n');
        
        stream.write('id: 1\n');
        stream.write('data: world\n\n');

        stream.write('event: close\n');
        stream.write('data: close\n\n');
    })

});

export default app;
```

## Client-Side Implementation

To use the power of SSE on the client side, a simple listener can be implemented:

```html
<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8">
    <title>SSE Example</title>
</head>

<body>
    <h1>SSE Example</h1>
    <div id="sse-data"></div>

    <script>
        const baseURL = "http://localhost:3000";
        const sseData = document.getElementById('sse-data');

        const eventSource = new EventSource(`${baseURL}/sse`);

        eventSource.onmessage = (event) => {
            sseData.innerHTML += event.data + '<br>';
        };

        eventSource.addEventListener("close", (event) => {
            console.log('Received "close" event. Closing connection...');
            eventSource.close();
        });

        eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
        };
    </script>
</body>

</html>
```

Visit the page in your browser, and watch as SSE seamlessly delivers server updates.

## Conclusion

Combining Hono with SSE opens doors to real-time, dynamic web applications. 

Consider exploring advanced Hono features and more about SSE:
- [Hono](https://hono.dev/) 
- [SSE (Server-Sent Events)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)

---
Receive my stream of work on:
- [X/Twitter](https://twitter.com/YanaelBarbier)
- [Threads](https://www.threads.net/@yanaelbarbier)
- [GitHub](https://github.com/st3w4r)

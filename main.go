package main

import (
    "context"
    "encoding/hex"
    "fmt"
    "github.com/drand/drand/client"
    "github.com/drand/drand/client/http"
    "log"
)

var urls = []string{
    "https://api.drand.sh",         // Official drand API server
    "https://drand.cloudflare.com", // Cloudflare's drand API server
}

var chainHash, _ = hex.DecodeString("8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce")

func main() {
    c, err := client.New(
        client.From(http.ForURLs(urls, chainHash)...),
        client.WithChainHash(chainHash),
    )
    if err != nil {
    log.Fatalf("error creating client: %v", err)
    }
    
    rand, err := c.Get(context.Background(), 0)
    if err != nil {
        log.Fatalf("Failed to get randomness: %v", err)
    }

    fmt.Printf("drand randomness: %s\n", hex.EncodeToString(rand.Randomness()))
}

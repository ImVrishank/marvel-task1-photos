#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include "model_data.h" // Your weights file

// =======================
// 1. CONFIGURATION
// =======================
const char* ssid = "Pixel 9a";
const char* password = "00000000";

// Camera Pins (AI Thinker Model)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WebServer server(80);

// =======================
// 2. AI INFERENCE ENGINE
// =======================
#define H 64
#define W 64

// Buffers in PSRAM (approx 130KB)
float* bufA; 
float* bufB;
float final_features[32];

// Helper to access Flash memory (PROGMEM) weights efficiently
float get_w(const float* weight_array, int index) {
    return pgm_read_float(weight_array + index);
}

void conv3x3(float* in, float* out, const float* w, const float* b, int inC, int outC, int curH, int curW) {
    for (int oc = 0; oc < outC; oc++) {
        float bias = get_w(b, oc);
        for (int i = 0; i < curH; i++) {
            for (int j = 0; j < curW; j++) {
                float sum = bias;
                for (int ic = 0; ic < inC; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int py = i + ky - 1; 
                            int px = j + kx - 1;
                            if (py >= 0 && py < curH && px >= 0 && px < curW) {
                                // Weight index calculation
                                int w_idx = oc*(inC*9) + ic*9 + ky*3 + kx;
                                sum += in[ic * curH * curW + py * curW + px] * get_w(w, w_idx);
                            }
                        }
                    }
                }
                out[oc * curH * curW + i * curW + j] = (sum > 0) ? sum : 0; // ReLU built-in
            }
        }
    }
}

void maxpool(float* in, float* out, int C, int curH, int curW) {
    int newH = curH / 2;
    int newW = curW / 2;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < newH; i++) {
            for (int j = 0; j < newW; j++) {
                float m = -10000.0;
                for (int ky = 0; ky < 2; ky++) {
                    for (int kx = 0; kx < 2; kx++) {
                        float val = in[c * curH * curW + (i * 2 + ky) * curW + (j * 2 + kx)];
                        if (val > m) m = val;
                    }
                }
                out[c * newH * newW + i * newW + j] = m;
            }
        }
    }
}

// Add these globals at the top if not present
float final_scores[25]; 
int predicted_class = 0;

void run_inference_engine() {
    Serial.println("Starting Inference...");
    unsigned long start = millis();

    // --- STEP 1: EXTRACT FEATURES (You already have this) ---
    // 1. Conv1
    conv3x3(bufA, bufB, layer_features_0_weight, layer_features_0_bias, 3, 8, 64, 64);
    // 2. MaxPool
    maxpool(bufB, bufA, 8, 64, 64);
    // 3. Conv2
    conv3x3(bufA, bufB, layer_features_3_weight, layer_features_3_bias, 8, 16, 32, 32);
    // 4. MaxPool
    maxpool(bufB, bufA, 16, 32, 32);
    // 5. Conv3
    conv3x3(bufA, bufB, layer_features_6_weight, layer_features_6_bias, 16, 32, 16, 16);

    // 6. Average Pool -> 32 Features
    for (int c = 0; c < 32; c++) {
        float sum = 0;
        for (int i = 0; i < 16*16; i++) sum += bufB[c * 256 + i];
        final_features[c] = sum / 256.0;
    }

    // --- STEP 2: CLASSIFIER (The missing piece!) ---
    // We have 25 output classes. 
    // We multiply our 32 features by the weights for each class.
    
    float max_score = -10000.0;
    
    for (int class_idx = 0; class_idx < 25; class_idx++) {
        float score = get_w(layer_classifier_bias, class_idx); // Start with bias
        
        for (int f = 0; f < 32; f++) {
            // Weights are flattened: [25 classes][32 features]
            int w_idx = class_idx * 32 + f; 
            score += final_features[f] * get_w(layer_classifier_weight, w_idx);
        }
        
        final_scores[class_idx] = score;
        
        // Find the winner (highest score)
        if (score > max_score) {
            max_score = score;
            predicted_class = class_idx;
        }
    }

    Serial.printf("Inference Time: %d ms\n", millis() - start);
    Serial.printf("Predicted Class: %d (Score: %.4f)\n", predicted_class, max_score);
}

// =======================
// 3. IMAGE PRE-PROCESSING
// =======================
// Convert ESP32 RGB565 (16-bit) to Planar Float (RRR..GGG..BBB)
// Also handles simple resizing/cropping to 64x64
void process_image(camera_fb_t * fb) {
    int startX = (fb->width - 64) / 2;
    int startY = (fb->height - 64) / 2;

    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            // Get pixel index in the large frame
            int idx = ((startY + y) * fb->width + (startX + x)) * 2;
            
            // RGB565 byte extraction
            uint8_t hb = fb->buf[idx];
            uint8_t lb = fb->buf[idx + 1];
            uint16_t pixel = (hb << 8) | lb;

            // Extract RGB
            float r = ((pixel >> 11) & 0x1F) / 31.0; // 0..1
            float g = ((pixel >> 5) & 0x3F) / 63.0;
            float b = (pixel & 0x1F) / 31.0;

            // Fill buffer Planar (CH, H, W)
            bufA[0 * 64 * 64 + y * 64 + x] = r; // R Channel
            bufA[1 * 64 * 64 + y * 64 + x] = g; // G Channel
            bufA[2 * 64 * 64 + y * 64 + x] = b; // B Channel
        }
    }
}

// =======================
// 4. WEB SERVER
// =======================
void handleRoot() {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "Camera Capture Failed");
        return;
    }

    // Process & Run AI
    process_image(fb);
    esp_camera_fb_return(fb); // Release memory early
    
    run_inference_engine();

    // Construct Response
    String html = "<h1>Smart AC AI Core</h1>";
    
    html += "<h2>Prediction: Class " + String(predicted_class) + "</h2>";
    
    html += "<h3>Scores:</h3><p>[ ";
    for(int i=0; i<25; i++) {
        html += String(final_scores[i], 2);
        if (i < 24) html += ", ";
    }
    html += " ]</p>";
    
    html += "<p><em>(Feature vector hidden)</em></p>";
    html += "<p><a href='/'>Refresh / Detect Again</a></p>";

    server.send(200, "text/html", html);
}

void setup() {
    Serial.begin(115200);

    // Alloc Buffers in PSRAM
    bufA = (float*)ps_malloc(8 * 64 * 64 * sizeof(float));
    bufB = (float*)ps_malloc(16 * 32 * 32 * sizeof(float));

    if(bufA == NULL || bufB == NULL) {
        Serial.println("PSRAM Malloc Failed! Enable PSRAM in Tools menu.");
        while(1);
    }

    // Camera Init
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565; // Crucial for AI processing
    config.frame_size = FRAMESIZE_QVGA;     // 320x240 (we crop to 64x64)
    config.jpeg_quality = 12;
    config.fb_count = 1;

    esp_camera_init(&config);

    // WiFi Init
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.print("Connected! IP: ");
    Serial.println(WiFi.localIP());

    server.on("/", handleRoot);
    server.begin();
    Serial.println("Web Server Started");
}

void loop() {
    server.handleClient();
}
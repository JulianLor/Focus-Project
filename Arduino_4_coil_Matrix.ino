#include <math.h>

// Constants
const float Hz = 1;             // Frequency of the AC signal
const float I_max = 64.0;         // Maximum amplitude for PWM (0-255 for 8-bit PWM)
const float rot_freq = 10;        // Rotation frequency
const float phi = PI / 2;    // Example phase angle adjustment in radians (30 degrees)

const float M[4][3] = {
    {1/(2 * sin(phi)), 0, 1/(4 * cos(phi))},
    {-1/(2 * sin(phi)), 0, 1/(4 * cos(phi))},
    {0, 1/(2 * sin(phi)), 1/(4 * cos(phi))},
    {0, -1/(2 * sin(phi)), 1/(4 * cos(phi))}
  };

// H-bridge PWM pins for each coil
const int pwmPin1A = 10, pwmPinINA1 = 12, pwmPinINB1 = 9;
const int pwmPin2A = 11, pwmPinINA2 = 13, pwmPinINB2 = 8;
const int pwmPin3A = 6, pwmPinINA3 = 7, pwmPinINB3 = 3;
const int pwmPin4A = 5, pwmPinINA4 = 4, pwmPinINB4 = 2;

void setup() {
    pinMode(pwmPin1A, OUTPUT); pinMode(pwmPinINA1, OUTPUT); pinMode(pwmPinINB1, OUTPUT);
    pinMode(pwmPin2A, OUTPUT); pinMode(pwmPinINA2, OUTPUT); pinMode(pwmPinINB2, OUTPUT);
    pinMode(pwmPin3A, OUTPUT); pinMode(pwmPinINA3, OUTPUT); pinMode(pwmPinINB3, OUTPUT);
    pinMode(pwmPin4A, OUTPUT); pinMode(pwmPinINA4, OUTPUT); pinMode(pwmPinINB4, OUTPUT);

    Serial.begin(9600); // For debugging
}

void MatrixMultiplication(float vector[3], float current[4]) {
    for (int i = 0; i < 4; i++) {
        current[i] = 0;  // Initialize result row
        for (int j = 0; j < 3; j++) {
            current[i] += M[i][j] * vector[j];
          }
      }
}

void rotateVector(float vector[3], char axis, float theta, float result[3]) {
    float rotationMatrix[3][3];

    // Choose the correct rotation matrix
    if (axis == 'x') {
        float cosT = cos(theta);
        float sinT = sin(theta);
        float tempMatrix[3][3] = {
            {1, 0, 0},
            {0, cosT, -sinT},
            {0, sinT, cosT}
        };
        memcpy(rotationMatrix, tempMatrix, sizeof(tempMatrix));
    } 
    else if (axis == 'y') {
        float cosT = cos(theta);
        float sinT = sin(theta);
        float tempMatrix[3][3] = {
            {cosT, 0, sinT},
            {0, 1, 0},
            {-sinT, 0, cosT}
        };
        memcpy(rotationMatrix, tempMatrix, sizeof(tempMatrix));
    } 
    else (axis == 'z') {
        float cosT = cos(theta);
        float sinT = sin(theta);
        float tempMatrix[3][3] = {
            {cosT, -sinT, 0},
            {sinT, cosT, 0},
            {0, 0, 1}
        };
        memcpy(rotationMatrix, tempMatrix, sizeof(tempMatrix));
    } 

    // Perform the matrix-vector multiplication
    for (int i = 0; i < 3; i++) {
        result[i] = 0;
        for (int j = 0; j < 3; j++) {
            result[i] += rotationMatrix[i][j] * vector[j];
        }
    }
}

void get_t_variables(float t, float t_state, float theta) {
    t_state = fmod(t, rot_freq);
    float t_theta = fmod(t_state, rot_freq / 3);
    theta = t_theta * PI / (2 * rot_freq / 3);
}

void get_v_rot(float v_rot[3], float t) {
    float x = sin(Hz * t * 2 * PI);
    float y = sin(Hz * t * 2 * PI);
    float z = 0;

    v_rot[3] = {x, y, z};
}

void get_b_vec(float v_rot[3], float t_state, float b_vec[3]) {
    // axis of rotation sweeps from z to x axis by pi/2
    if (t_state < rot_freq / 3) {
        rotateVector(direction, 'y', theta, b_vec);
    }

    // axis of rotation sweeps from x to y axis by pi/2
    else if (t_state < (2 * rot_freq / 3)) {
        rotateVector(direction, 'z', theta, b_vec);
    }

    // axis of rotation sweeps from x to y axis by pi/2
    else {
        rotateVector(direction, 'x', theta, b_vec);
    }
}

void IN_pins(float current[4]) {
    int pwmPinINA1_val, pwmPinINB1_val, pwmPinINA2_val, pwmPinINB2_val, pwmPinINA3_val, pwmPinINB3_val, pwmPinINA4_val, pwmPinINB4_val;

    // Enabler pins for Coil 1 
    if (current[0] < 0) {
        pwmPinINA1_val = HIGH;
        pwmPinINB1_val = LOW;
    }
    else {
        pwmPinINA1_val = LOW;
        pwmPinINB1_val = HIGH;
    }

    // Enabler pins for Coil 2
    if (current[1] < 0) {
        pwmPinINA2_val = HIGH;
        pwmPinINB2_val = LOW;
    }
    else {
        pwmPinINA2_val = LOW;
        pwmPinINB2_val = HIGH;
    }

    // Enabler pins for Coil 3
    if (current[2] < 0) {
        pwmPinINA3_val = HIGH;
        pwmPinINB3_val = LOW;
    }
    else {
        pwmPinINA3_val = LOW;
        pwmPinINB3_val = HIGH;
    }

    // Enabler pins for Coil 4
    if (current[3] < 0) {
        pwmPinINA4_val = HIGH;
        pwmPinINB4_val = LOW;
    }
    else {
        pwmPinINA4_val = LOW;
        pwmPinINB4_val = HIGH;
    }

    digitalWrite(pwmPinINA1, pwmPinINA1_val);
    digitalWrite(pwmPinINB1, pwmPinINB1_val);

    digitalWrite(pwmPinINA2, pwmPinINA2_val);
    digitalWrite(pwmPinINB2, pwmPinINB2_val);

    digitalWrite(pwmPinINA3, pwmPinINA3_val);
    digitalWrite(pwmPinINB3, pwmPinINB3_val);

    digitalWrite(pwmPinINA4, pwmPinINA4_val);
    digitalWrite(pwmPinINB4, pwmPinINB4_val);
}

void abs_val_current(float current[4], float current_abs[4]) {
    float current_abs[0] = abs(current[0]);
    float current_abs[1] = abs(current[1]);
    float current_abs[2] = abs(current[2]);
    float current_abs[3] = abs(current[3]);
}

void print_current(float current[4]) {
    Serial.print("Current1 output:");
    Serial.println(current[0]);
    Serial.print("Current2 output:");
    Serial.println(current[1]);
    Serial.print("Current3 output:");
    Serial.println(current3[2];
    Serial.print("Current4 output:");
    Serial.println(current[3]);
}

void PWM_pins(float current_abs[4]) {
    int pwmValue1 = int(current_abs[0] * I_max); // Scale sine to 0-255
    int pwmValue2 = int(current_abs[1] * I_max); // Scale cosine to 0-255
    int pwmValue3 = int(current_abs[2] * I_max); // Scale sine to 0-255
    int pwmValue4 = int(current_abs[3] * I_max); // Scale cosine to 0-255

    analogWrite(pwmPin1A, pwmValue1);  
    analogWrite(pwmPin2A, pwmValue2);
    analogWrite(pwmPin3A, pwmValue3);  
    analogWrite(pwmPin4A, pwmValue4);
}

void loop() {
    float t = millis() / 1000.0; // Time in seconds
    
    float t_state, theta;
    get_t_variables(t, theta);

    float v_rot[3];
    get_v_rot(v_rot, t);

    float v_rot[3], b_vec[3];
    get_b_vec(v_rot[3], t_state, b_vec);

    float current[4]
    MatrixMultiplication(b_vec, current);

    IN_pins(current);

    float current_abs[4];
    abs_val_current(current, current_abs);

    print_current(current);

    PWM_pins(current_abs[4]);

    delay(5);  // Short delay to manage loop frequency
  }
  

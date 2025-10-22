package com.example.ecoscan; // Mude "com.example.ecoscan" se o seu pacote for diferente

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    // Constantes
    private static final String TAG = "EcoScanApp";
    private static final String MODEL_FILE = "yolov8.tflite"; // Nome do seu modelo TFLite
    private static final float CONFIDENCE_THRESHOLD = 0.5f; // Limite de confiança (50%)

    // Componentes da Tela
    private ImageView imageView;
    private Button buttonCamera;
    private Button buttonGallery;
    private TextView textViewResult;

    // Detector de Objetos do TensorFlow Lite
    private ObjectDetector objectDetector;

    // Activity Result Launchers (a forma moderna de lidar com resultados de intents)
    private ActivityResultLauncher<Intent> cameraLauncher;
    private ActivityResultLauncher<String> galleryLauncher;
    private ActivityResultLauncher<String> permissionLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Inicializa os componentes da tela
        imageView = findViewById(R.id.imageView);
        buttonCamera = findViewById(R.id.buttonCamera);
        buttonGallery = findViewById(R.id.buttonGallery);
        textViewResult = findViewById(R.id.textViewResult);

        // Configura o detector de objetos
        setupObjectDetector();

        // Configura os launchers
        setupLaunchers();

        // Configura os cliques dos botões
        buttonCamera.setOnClickListener(v -> checkCameraPermissionAndOpenCamera());
        buttonGallery.setOnClickListener(v -> openGallery());
    }

    private void setupObjectDetector() {
        try {
            // Configurações do detector
            ObjectDetector.ObjectDetectorOptions options = ObjectDetector.ObjectDetectorOptions.builder()
                    .setMaxResults(1) // Queremos apenas o resultado mais provável
                    .setScoreThreshold(CONFIDENCE_THRESHOLD)
                    .build();

            // Cria o detector a partir do modelo nos assets
            objectDetector = ObjectDetector.createFromFileAndOptions(this, MODEL_FILE, options);

        } catch (IOException e) {
            Log.e(TAG, "Erro ao inicializar o modelo TFLite.", e);
            Toast.makeText(this, "Não foi possível carregar o modelo.", Toast.LENGTH_SHORT).show();
        }
    }

    private void setupLaunchers() {
        // Launcher para permissão de câmera
        permissionLauncher = registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
            if (isGranted) {
                openCamera();
            } else {
                Toast.makeText(this, "Permissão da câmera negada.", Toast.LENGTH_SHORT).show();
            }
        });

        // Launcher para a câmera
        cameraLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                Bundle extras = result.getData().getExtras();
                Bitmap imageBitmap = (Bitmap) extras.get("data");
                if (imageBitmap != null) {
                    detectObjects(imageBitmap);
                }
            }
        });

        // Launcher para a galeria
        galleryLauncher = registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
            if (uri != null) {
                try {
                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    detectObjects(bitmap);
                } catch (IOException e) {
                    Log.e(TAG, "Erro ao carregar imagem da galeria.", e);
                }
            }
        });
    }

    private void checkCameraPermissionAndOpenCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void openCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraLauncher.launch(takePictureIntent);
    }

    private void openGallery() {
        galleryLauncher.launch("image/*");
    }

    private void detectObjects(Bitmap bitmap) {
        if (objectDetector == null) {
            Toast.makeText(this, "Detector não foi inicializado.", Toast.LENGTH_SHORT).show();
            return;
        }

        // Converte o Bitmap para o formato que o TFLite entende
        TensorImage tensorImage = TensorImage.fromBitmap(bitmap);

        // Roda a detecção
        List<Detection> results = objectDetector.detect(tensorImage);

        // Processa e exibe os resultados
        displayDetectionResult(results, bitmap);
    }

    private void displayDetectionResult(List<Detection> detections, Bitmap originalBitmap) {
        if (detections == null || detections.isEmpty()) {
            textViewResult.setText("Nenhum objeto reconhecido.\nTente uma foto com melhor iluminação ou enquadramento.");
            imageView.setImageBitmap(originalBitmap);
            return;
        }

        // Pega a detecção com maior probabilidade (já que configuramos maxResults=1)
        Detection bestDetection = detections.get(0);
        String label = bestDetection.getCategories().get(0).getLabel();
        float confidence = bestDetection.getCategories().get(0).getScore();
        String disposalInfo = getDisposalInfo(label);

        // Formata o texto do resultado
        String resultText = String.format(Locale.getDefault(),
                "Objeto: %s\nCerteza: %.1f%%\nDescarte: %s",
                label,
                confidence * 100,
                disposalInfo);
        textViewResult.setText(resultText);

        // Desenha a caixa de detecção na imagem
        Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5.0f);

        canvas.drawRect(bestDetection.getBoundingBox(), paint);
        imageView.setImageBitmap(mutableBitmap);
    }

    // AQUI VOCÊ DEVE PERSONALIZAR COM SUAS CLASSES E INSTRUÇÕES DE DESCARTE
    private String getDisposalInfo(String label) {
        switch (label.toLowerCase()) {
            case "garrafa pet":
            case "plastico":
                return "Lixo reciclável - PLÁSTICO (Amarelo)";
            case "papel":
            case "papelao":
                return "Lixo reciclável - PAPEL (Azul)";
            case "lata":
            case "metal":
                return "Lixo reciclável - METAL (Amarelo)";
            case "vidro":
                return "Lixo reciclável - VIDRO (Verde)";
            case "organico":
                return "Lixo orgânico / Compostagem";
            default:
                return "Lixo comum / Não reciclável";
        }
    }
}


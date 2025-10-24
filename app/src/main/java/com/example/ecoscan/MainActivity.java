package com.example.ecoscan; // Mude "com.example.ecoscan" se o seu pacote for diferente

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    // Constantes
    private static final String TAG = "EcoScanApp";
    private static final String MODEL_FILE = "best.tflite";
    private static final String LABEL_FILE = "labels.txt";
    private static final float CONFIDENCE_THRESHOLD = 0.2f;
    private static final int INPUT_SIZE = 640;

    // Componentes da Tela
    private ImageView imageView;
    private Button buttonCamera;
    private Button buttonGallery;
    private Button buttonAnalyze;
    private TextView textViewResult;

    // TensorFlow Lite
    private Interpreter interpreter;
    private List<String> labels = new ArrayList<>();
    private int inputWidth;
    private int inputHeight;
    private int outputNumClasses;
    private int outputNumProposals;

    // Processador de Imagem
    private ImageProcessor imageProcessor;

    // Launchers
    private ActivityResultLauncher<Intent> cameraLauncher;
    private ActivityResultLauncher<Intent> galleryLauncher;
    private ActivityResultLauncher<String> permissionLauncher;

    private Bitmap bitmapToAnalyze;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        buttonCamera = findViewById(R.id.buttonCamera);
        buttonGallery = findViewById(R.id.buttonGallery);
        buttonAnalyze = findViewById(R.id.buttonAnalyze);
        textViewResult = findViewById(R.id.textViewResult);

        // Assim que o app abrir, você verá o texto vermelho "[TESTE] Aguardando análise..."
        // que definimos no XML.

        setupLaunchers();

        try {
            loadLabels();
            loadModel();
            setupImageProcessor();
            Log.d(TAG, "TensorFlow Lite inicializado com sucesso.");
        } catch (IOException e) {
            Log.e(TAG, "Falha ao inicializar o TensorFlow Lite.", e);
            Toast.makeText(this, "Não foi possível carregar o modelo ou labels. Verifique assets.", Toast.LENGTH_LONG).show();
            textViewResult.setText("FALHA AO CARREGAR MODELO. Verifique o Logcat."); // Mostra erro
        }

        buttonCamera.setOnClickListener(v -> checkCameraPermissionAndOpenCamera());
        buttonGallery.setOnClickListener(v -> openGallery());

        buttonAnalyze.setOnClickListener(v -> analyzeImage());
        buttonAnalyze.setEnabled(false);
    }

    private void analyzeImage() {
        if (bitmapToAnalyze != null) {
            // ##### MUDANÇA 1: AVISA QUE ESTÁ ANALISANDO #####
            textViewResult.setText("Analisando...");
            Log.d(TAG, "Iniciando detecção...");
            detectObjects(bitmapToAnalyze);
        } else {
            Toast.makeText(this, "Selecione uma imagem da câmera ou galeria primeiro.", Toast.LENGTH_SHORT).show();
            textViewResult.setText("Nenhuma imagem selecionada para análise.");
        }
    }

    private void loadLabels() throws IOException {
        AssetManager assetManager = getAssets();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(LABEL_FILE)));
        String line;
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
        Log.d(TAG, "Labels carregados: " + labels.size());
        Log.d(TAG, "Labels: " + labels.toString());
    }

    private void loadModel() throws IOException {
        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(this, MODEL_FILE);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(tfliteModel, options);

        int[] inputShape = interpreter.getInputTensor(0).shape();
        inputWidth = inputShape[1];
        inputHeight = inputShape[2];

        int[] outputShape = interpreter.getOutputTensor(0).shape();
        outputNumClasses = outputShape[1] - 4;
        outputNumProposals = outputShape[2];

        Log.d(TAG, "Modelo carregado. Entrada: " + inputWidth + "x" + inputHeight);
        Log.d(TAG, "Saída: Classes=" + outputNumClasses + ", Propostas=" + outputNumProposals);

        if (outputNumClasses != labels.size()) {
            String errorMsg = "Erro: Modelo espera " + outputNumClasses + " classes, mas " + LABEL_FILE + " tem " + labels.size() + " classes.";
            Log.e(TAG, errorMsg);
            Toast.makeText(this, "Erro: Incompatibilidade entre modelo e labels.txt", Toast.LENGTH_LONG).show();
            textViewResult.setText(errorMsg); // Mostra erro
        }
    }

    private void setupImageProcessor() {
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0f, 255f))
                .build();
    }

    private void setupLaunchers() {
        permissionLauncher = registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
            if (isGranted) openCamera();
            else Toast.makeText(this, "Permissão da câmera negada.", Toast.LENGTH_SHORT).show();
        });

        cameraLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                Bundle extras = result.getData().getExtras();
                if (extras != null) {
                    Bitmap imageBitmap = (Bitmap) extras.get("data");
                    if (imageBitmap != null) {
                        bitmapToAnalyze = imageBitmap;
                        imageView.setImageBitmap(bitmapToAnalyze);
                        // ##### MUDANÇA 2: AVISA QUE IMAGEM ESTÁ PRONTA #####
                        textViewResult.setText("Imagem carregada. Clique em 'Analisar'.");
                        buttonAnalyze.setEnabled(true);
                    }
                }
            }
        });

        galleryLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                Uri imageUri = result.getData().getData();
                if (imageUri != null) {
                    try {
                        Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                        bitmapToAnalyze = bitmap;
                        imageView.setImageBitmap(bitmapToAnalyze);
                        // ##### MUDANÇA 3: AVISA QUE IMAGEM ESTÁ PRONTA #####
                        textViewResult.setText("Imagem carregada. Clique em 'Analisar'.");
                        buttonAnalyze.setEnabled(true);
                    } catch (IOException e) {
                        Log.e(TAG, "Erro ao carregar imagem da galeria.", e);
                        textViewResult.setText("Erro ao carregar imagem da galeria.");
                    }
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
        Intent pickIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryLauncher.launch(pickIntent);
    }

    private void detectObjects(Bitmap bitmap) {
        if (interpreter == null || labels.isEmpty()) {
            String errorMsg = "Detector não foi inicializado. Verifique o Logcat.";
            Toast.makeText(this, errorMsg, Toast.LENGTH_SHORT).show();
            Log.e(TAG, errorMsg + " Verifique se o modelo e labels foram carregados.");
            textViewResult.setText(errorMsg); // Mostra erro
            return;
        }

        TensorImage tensorImage = TensorImage.fromBitmap(bitmap);
        tensorImage = imageProcessor.process(tensorImage);
        ByteBuffer inputBuffer = tensorImage.getBuffer();

        float[][][] outputArray = new float[1][4 + outputNumClasses][outputNumProposals];

        Log.d(TAG, "Rodando o modelo...");
        try {
            interpreter.run(inputBuffer, outputArray);
            Log.d(TAG, "Modelo executado com sucesso.");
        } catch (Exception e) {
            Log.e(TAG, "Erro na execução do modelo TFLite.", e);
            Toast.makeText(this, "Erro na execução do modelo. Formato de entrada/saída incorreto.", Toast.LENGTH_LONG).show();
            // ##### MUDANÇA 4: MOSTRA ERRO DE EXECUÇÃO NA TELA #####
            textViewResult.setText("Erro ao executar modelo: " + e.getMessage());
            return;
        }

        List<Detection> detections = postProcessYolo(outputArray[0], bitmap.getWidth(), bitmap.getHeight());

        Log.d(TAG, "Detecções pós-processadas: " + detections.size());
        displayDetectionResult(detections, bitmap);
    }

    // Classe interna para guardar uma detecção
    private static class Detection {
        RectF boundingBox;
        String label;
        float confidence;

        Detection(RectF boundingBox, String label, float confidence) {
            this.boundingBox = boundingBox;
            this.label = label;
            this.confidence = confidence;
        }
    }

    private List<Detection> postProcessYolo(float[][] output, int originalWidth, int originalHeight) {
        float[][] transposedOutput = new float[outputNumProposals][4 + outputNumClasses];
        for (int i = 0; i < 4 + outputNumClasses; i++) {
            for (int j = 0; j < outputNumProposals; j++) {
                transposedOutput[j][i] = output[i][j];
            }
        }

        List<Detection> allDetections = new ArrayList<>();

        for (int i = 0; i < outputNumProposals; i++) {
            float[] proposal = transposedOutput[i];

            int bestClassIndex = -1;
            float maxScore = 0.0f;
            for (int j = 4; j < 4 + outputNumClasses; j++) {
                if (proposal[j] > maxScore) {
                    maxScore = proposal[j];
                    bestClassIndex = j - 4;
                }
            }

            if (maxScore > CONFIDENCE_THRESHOLD) {
                float cx = proposal[0];
                float cy = proposal[1];
                float w = proposal[2];
                float h = proposal[3];

                float left = cx - (w / 2f);
                float top = cy - (h / 2f);
                float right = cx + (w / 2f);
                float bottom = cy + (h / 2f);

                String label = (bestClassIndex >= 0 && bestClassIndex < labels.size()) ? labels.get(bestClassIndex) : "Desconhecido";

                RectF boundingBox = new RectF(left / INPUT_SIZE, top / INPUT_SIZE, right / INPUT_SIZE, bottom / INPUT_SIZE);
                allDetections.add(new Detection(boundingBox, label, maxScore));
            }
        }

        return nonMaxSuppression(allDetections);
    }

    private List<Detection> nonMaxSuppression(List<Detection> allDetections) {
        List<Detection> nmsList = new ArrayList<>();
        float IOU_THRESHOLD = 0.45f;

        allDetections.sort(Comparator.comparingDouble((Detection d) -> d.confidence).reversed());

        for (Detection detection : allDetections) {
            boolean keep = true;
            for (Detection nmsDetection : nmsList) {
                float iou = calculateIoU(detection.boundingBox, nmsDetection.boundingBox);
                if (iou > IOU_THRESHOLD) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                nmsList.add(detection);
            }
        }
        return nmsList;
    }

    private float calculateIoU(RectF boxA, RectF boxB) {
        float xA = Math.max(boxA.left, boxB.left);
        float yA = Math.max(boxA.top, boxB.top);
        float xB = Math.min(boxA.right, boxB.right);
        float yB = Math.min(boxA.bottom, boxB.bottom);

        float interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        float boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top);
        float boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top);
        float unionArea = (boxAArea + boxBArea - interArea);

        return interArea / unionArea;
    }


    private void displayDetectionResult(List<Detection> detections, Bitmap originalBitmap) {
        Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.parseColor("#4CAF50")); // Verde EcoScan
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(Math.max(mutableBitmap.getWidth(), mutableBitmap.getHeight()) / 100f);
        paint.setTextSize(Math.max(mutableBitmap.getWidth(), mutableBitmap.getHeight()) / 25f);
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.FILL_AND_STROKE);

        if (detections.isEmpty()) {
            // ##### MUDANÇA 5: AVISA QUE NADA FOI ACHADO #####
            Log.w(TAG, "Nenhuma detecção válida encontrada (acima do threshold).");
            textViewResult.setText("Nenhum objeto reconhecido.\nTente uma foto com melhor iluminação ou enquadramento.");
            imageView.setImageBitmap(originalBitmap); // Mostra a original se nada for achado
            return;
        }

        Detection bestDetection = detections.get(0);
        String label = bestDetection.label;
        float confidence = bestDetection.confidence;
        String disposalInfo = getDisposalInfo(label);

        String resultText = String.format(Locale.getDefault(),
                "Objeto: %s\nCerteza: %.1f%%\nDescarte: %s",
                label,
                confidence * 100,
                disposalInfo);

        // ##### MUDANÇA 6: EXIBE O RESULTADO FINAL #####
        // O texto continuará vermelho, como definido no XML.
        textViewResult.setText(resultText);
        Log.i(TAG, "Resultado exibido: " + resultText);

        // Desenha a caixa de detecção na imagem
        RectF scaledBox = new RectF(
                bestDetection.boundingBox.left * originalBitmap.getWidth(),
                bestDetection.boundingBox.top * originalBitmap.getHeight(),
                bestDetection.boundingBox.right * originalBitmap.getWidth(),
                bestDetection.boundingBox.bottom * originalBitmap.getHeight()
        );

        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.parseColor("#4CAF50"));
        canvas.drawRect(scaledBox, paint);

        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.parseColor("#4CAF50"));
        canvas.drawText(
                String.format(Locale.getDefault(), "%s (%.1f%%)", label, confidence * 100),
                scaledBox.left + 10,
                scaledBox.top - 10,
                paint
        );

        imageView.setImageBitmap(mutableBitmap);
    }

    // RELEMBRANDO: Este método foi corrigido na última versão e deve estar correto
    // com o seu labels.txt
    private String getDisposalInfo(String label) {
        switch (label.toLowerCase(Locale.ROOT)) {
            case "plastic":
                return "Lixo reciclável - PLÁSTICO (Vermelho)";
            case "paper":
                return "Lixo reciclável - PAPEL (Azul)";
            case "cardboard":
                return "Lixo reciclável - PAPELÃO (Azul)";
            case "metal":
                return "Lixo reciclável - METAL (Amarelo)";
            case "glass":
                return "Lixo reciclável - VIDRO (Verde)";
            case "biodegradable":
                return "Lixo orgânico / Compostagem (Marrom)";
            default:
                return "Lixo comum / Não reciclável.";
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (interpreter != null) {
            interpreter.close();
        }
    }
}
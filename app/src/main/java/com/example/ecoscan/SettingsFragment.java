package com.example.ecoscan;

import android.Manifest;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog; // <<< ADICIONADO: Import para o Dialog
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.fragment.app.Fragment;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class SettingsFragment extends Fragment {

    // --- Suas variáveis (exatamente como ScanFragment) ---
    private static final String TAG = "EcoScanApp_Experiment";
    private static final String MODEL_FILE = "best.tflite";
    private static final String LABEL_FILE = "labels.txt";
    private static final float CONFIDENCE_THRESHOLD = 0.2f;
    private static final int INPUT_SIZE = 640;

    private ImageView imageView;
    private Button buttonCamera;
    private Button buttonGallery;
    private Button buttonAnalyze;
    private TextView textViewResult;

    private Interpreter interpreter;
    private List<String> labels = new ArrayList<>();
    private int inputWidth;
    private int inputHeight;
    private int outputNumClasses;
    private int outputNumProposals;

    private ImageProcessor imageProcessor;

    // Launchers
    private ActivityResultLauncher<Intent> cameraLauncher;
    private ActivityResultLauncher<Intent> galleryLauncher;
    private ActivityResultLauncher<String> permissionLauncher;

    private Bitmap bitmapToAnalyze;
    private Uri cameraImageUri;

    // <<< ADICIONADO: Variável para guardar o bitmap com as detecções
    private Bitmap bitmapWithDetections;

    // --- Métodos de Ciclo de Vida do Fragment ---

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setupLaunchers();
        try {
            loadLabels();
            loadModel();
            setupImageProcessor();
            Log.d(TAG, "TensorFlow Lite (Experimento) inicializado.");
        } catch (IOException e) {
            Log.e(TAG, "Falha ao inicializar o TensorFlow Lite.", e);
            Toast.makeText(requireContext(), "Não foi possível carregar o modelo ou labels.", Toast.LENGTH_LONG).show();
        }
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_settings, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        imageView = view.findViewById(R.id.imageView);
        buttonCamera = view.findViewById(R.id.buttonCamera);
        buttonGallery = view.findViewById(R.id.buttonGallery);
        buttonAnalyze = view.findViewById(R.id.buttonAnalyze);
        textViewResult = view.findViewById(R.id.textViewResult);
        textViewResult.setText("Modo Experimento: Mostra todas as detecções.");
        buttonCamera.setOnClickListener(v -> checkCameraPermissionAndOpenCamera());
        buttonGallery.setOnClickListener(v -> openGallery());
        buttonAnalyze.setOnClickListener(v -> analyzeImage());
        buttonAnalyze.setEnabled(false);

        // <<< ADICIONADO: Listener de clique no ImageView
        imageView.setOnClickListener(v -> {
            // Só abre o dialog se tivermos um bitmap de resultado
            if (bitmapWithDetections != null) {
                showImageInDialog(bitmapWithDetections);
            }
        });
    }

    // --- Lógica de Análise ---

    private void analyzeImage() {
        if (bitmapToAnalyze != null) {
            textViewResult.setText("Analisando...");
            Log.d(TAG, "Iniciando detecção (Experimento)...");
            // Mostra a imagem original brevemente enquanto analisa
            imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
            imageView.setImageBitmap(bitmapToAnalyze);
            detectObjects(bitmapToAnalyze);
        } else {
            Toast.makeText(requireContext(), "Selecione uma imagem da câmera ou galeria primeiro.", Toast.LENGTH_SHORT).show();
            textViewResult.setText("Nenhuma imagem selecionada para análise.");
        }
    }

    // --- Correção do Hardware Bitmap ---
    private Bitmap loadBitmapFromUri(Uri uri) throws IOException {
        ContentResolver resolver = requireContext().getContentResolver();
        ImageDecoder.Source source = ImageDecoder.createSource(resolver, uri);

        return ImageDecoder.decodeBitmap(source, (decoder, info, s) -> {
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            decoder.setMutableRequired(true);
        });
    }

    // --- Continuação dos seus métodos (Launcher, Permissão, TFLite) ---

    private Uri createImageUri() {
        File imagePath = new File(requireContext().getCacheDir(), "images");
        if (!imagePath.exists()) imagePath.mkdirs();

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        File newFile = new File(imagePath, "IMG_" + timeStamp + ".jpg");

        return FileProvider.getUriForFile(requireContext(), requireContext().getPackageName() + ".provider", newFile);
    }

    private void setupLaunchers() {
        permissionLauncher = registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
            if (isGranted) openCamera();
            else Toast.makeText(requireContext(), "Permissão da câmera negada.", Toast.LENGTH_SHORT).show();
        });

        cameraLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == android.app.Activity.RESULT_OK) {
                if (cameraImageUri != null) {
                    try {
                        bitmapToAnalyze = loadBitmapFromUri(cameraImageUri);
                        bitmapWithDetections = null; // <<< MODIFICADO: Reseta o bitmap de análise
                        imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
                        imageView.setImageBitmap(bitmapToAnalyze);
                        textViewResult.setText("Imagem carregada. Clique em 'Analisar'.");
                        buttonAnalyze.setEnabled(true);
                    } catch (IOException e) {
                        Log.e(TAG, "Erro ao carregar imagem da câmera (Uri).", e);
                        textViewResult.setText("Erro ao carregar foto.");
                    }
                }
            }
        });

        galleryLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == android.app.Activity.RESULT_OK && result.getData() != null) {
                Uri imageUri = result.getData().getData();
                if (imageUri != null) {
                    try {
                        bitmapToAnalyze = loadBitmapFromUri(imageUri);
                        bitmapWithDetections = null; // <<< MODIFICADO: Reseta o bitmap de análise
                        imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
                        imageView.setImageBitmap(bitmapToAnalyze);
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
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void openCamera() {
        cameraImageUri = createImageUri();
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, cameraImageUri);
        cameraLauncher.launch(takePictureIntent);
    }

    private void openGallery() {
        Intent pickIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryLauncher.launch(pickIntent);
    }

    // --- Métodos de TFLite (Semelhantes) ---

    private void loadLabels() throws IOException {
        AssetManager assetManager = requireContext().getAssets();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(LABEL_FILE)));
        String line;
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
    }

    private void loadModel() throws IOException {
        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(requireContext(), MODEL_FILE);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(tfliteModel, options);
        int[] inputShape = interpreter.getInputTensor(0).shape();
        inputWidth = inputShape[1];
        inputHeight = inputShape[2];
        int[] outputShape = interpreter.getOutputTensor(0).shape();
        outputNumClasses = outputShape[1] - 4;
        outputNumProposals = outputShape[2];
    }

    private void setupImageProcessor() {
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0f, 255f))
                .build();
    }

    private void detectObjects(Bitmap bitmap) {
        if (interpreter == null || labels.isEmpty()) {
            Toast.makeText(requireContext(), "Detector não foi inicializado.", Toast.LENGTH_SHORT).show();
            return;
        }

        TensorImage tensorImage = TensorImage.fromBitmap(bitmap);
        tensorImage = imageProcessor.process(tensorImage);
        ByteBuffer inputBuffer = tensorImage.getBuffer();

        float[][][] outputArray = new float[1][4 + outputNumClasses][outputNumProposals];

        try {
            interpreter.run(inputBuffer, outputArray);
        } catch (Exception e) {
            Log.e(TAG, "Erro na execução do modelo TFLite.", e);
            // CORREÇÃO: Adicionando os parênteses () para chamar o método show()
            Toast.makeText(requireContext(), "Erro na execução do modelo.", Toast.LENGTH_LONG).show();
            return;
        }

        List<Detection> detections = postProcessYolo(outputArray[0], bitmap.getWidth(), bitmap.getHeight());
        displayAllDetections(detections, bitmap);
    }

    // Classe interna (igual)
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

    // MÉTODO CORRIGIDO (da sua solicitação anterior)
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

            // Encontra a classe com maior pontuação
            for (int j = 4; j < 4 + outputNumClasses; j++) {
                if (proposal[j] > maxScore) {
                    maxScore = proposal[j];
                    bestClassIndex = j - 4;
                }
            }

            if (maxScore > CONFIDENCE_THRESHOLD) {
                // proposal[0] a proposal[3] são (cx, cy, w, h) já normalizados [0, 1]
                float cx = proposal[0];
                float cy = proposal[1];
                float w = proposal[2];
                float h = proposal[3];

                // Calcula as coordenadas normalizadas [0, 1] para (left, top, right, bottom)
                float left = cx - (w / 2f);
                float top = cy - (h / 2f);
                float right = cx + (w / 2f);
                float bottom = cy + (h / 2f);

                String label = (bestClassIndex >= 0 && bestClassIndex < labels.size()) ? labels.get(bestClassIndex) : "Desconhecido";

                // CRIA O RectF com as coordenadas normalizadas [0, 1]
                RectF boundingBox = new RectF(left, top, right, bottom);

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
            if (keep) nmsList.add(detection);
        }
        return nmsList;
    }

    private float calculateIoU(RectF boxA, RectF boxB) {
        float xA = Math.max(boxA.left, boxB.left), yA = Math.max(boxA.top, boxB.top);
        float xB = Math.min(boxA.right, boxB.right), yB = Math.min(boxA.bottom, boxB.bottom);
        float interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        float boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top);
        float boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top);
        float unionArea = (boxAArea + boxBArea - interArea);
        return interArea / unionArea;
    }

    // --- MÉTODO DE RESULTADO (Experimento) ---

    private void displayAllDetections(List<Detection> detections, Bitmap originalBitmap) {
        if (detections.isEmpty()) {
            textViewResult.setText("Nenhum objeto reconhecido. Tente novamente.");
            this.bitmapWithDetections = null; // <<< MODIFICADO: Reseta o bitmap de análise
            imageView.setImageBitmap(originalBitmap);
            return;
        }

        // <<< MODIFICADO: Salva o bitmap resultante na variável da classe
        this.bitmapWithDetections = drawAllDetections(originalBitmap, detections);

        imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        imageView.setImageBitmap(this.bitmapWithDetections); // Mostra o bitmap salvo

        String resultText = String.format(Locale.US, "Encontrados %d objetos.", detections.size());
        textViewResult.setText(resultText);
    }

    // (O método drawAllDetections permanece exatamente o mesmo que você já tinha)
    private Bitmap drawAllDetections(Bitmap originalBitmap, List<Detection> detections) {
        Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        Paint boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(Math.max(mutableBitmap.getWidth(), mutableBitmap.getHeight()) / 120f);

        // Configuração de texto e fundo de texto
        float textScaleFactor = Math.max(mutableBitmap.getWidth(), mutableBitmap.getHeight()) / 600f; // Fator para tornar o texto responsivo
        float textSize = 30 * textScaleFactor; // Tamanho base do texto
        float padding = 8 * textScaleFactor; // Padding base

        Paint textPaint = new Paint();
        textPaint.setTextSize(textSize);
        textPaint.setColor(Color.WHITE);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextAlign(Paint.Align.LEFT); // Alinhar o texto à esquerda

        Paint textBgPaint = new Paint();
        textBgPaint.setStyle(Paint.Style.FILL);

        int[] colors = new int[]{
                Color.parseColor("#D32F2F"),
                Color.parseColor("#1976D2"),
                Color.parseColor("#FBC02D"),
                Color.parseColor("#388E3C"),
                Color.parseColor("#5D4037"),
                Color.parseColor("#FF5722")
        };

        int colorIndex = 0;

        // Obter métricas de texto para cálculo de fundo
        Paint.FontMetrics fm = textPaint.getFontMetrics();
        float textHeight = fm.descent - fm.ascent;


        for (Detection detection : detections) {
            int color = colors[colorIndex % colors.length];
            boxPaint.setColor(color);
            textBgPaint.setColor(color);
            colorIndex++;

            String label = String.format(Locale.US, "%s: %.1f%%",
                    detection.label,
                    detection.confidence * 100);

            // 1. Calcular a caixa de detecção escalada
            RectF scaledBox = new RectF(
                    detection.boundingBox.left * originalBitmap.getWidth(),
                    detection.boundingBox.top * originalBitmap.getHeight(),
                    detection.boundingBox.right * originalBitmap.getWidth(),
                    detection.boundingBox.bottom * originalBitmap.getHeight()
            );

            // Desenhar a caixa de detecção
            canvas.drawRect(scaledBox, boxPaint);

            // 2. Calcular a caixa de fundo do texto (label)
            float labelWidth = textPaint.measureText(label);

            // Posição vertical da caixa de texto (acima da scaledBox)
            float textBgTop = scaledBox.top - textHeight - (padding * 2);
            float textBgBottom = scaledBox.top; // A base da caixa de texto encosta no topo da scaledBox

            // Ajuste se a caixa de texto sair do limite superior da imagem
            if (textBgTop < 0) {
                textBgTop = scaledBox.bottom; // Coloca abaixo se não couber acima
                textBgBottom = scaledBox.bottom + textHeight + (padding * 2);
            }

            RectF textBgRect = new RectF();
            textBgRect.left = scaledBox.left;
            textBgRect.top = textBgTop;
            textBgRect.right = scaledBox.left + labelWidth + (padding * 2);
            textBgRect.bottom = textBgBottom;

            // Desenhar o fundo do texto
            canvas.drawRect(textBgRect, textBgPaint);

            // 3. Desenhar o texto (label)
            // O texto é desenhado com a base da linha na coordenada Y.
            float textY = textBgRect.top - fm.ascent + padding;

            canvas.drawText(label,
                    textBgRect.left + padding,
                    textY,
                    textPaint);
        }

        return mutableBitmap;
    }

    // <<< ADICIONADO: Novo método para mostrar a imagem em um Dialog
    private void showImageInDialog(Bitmap bitmap) {
        if (getContext() == null) return;

        // 1. Cria um ImageView programaticamente para colocar no Dialog
        ImageView dialogImageView = new ImageView(getContext());
        dialogImageView.setImageBitmap(bitmap);
        dialogImageView.setScaleType(ImageView.ScaleType.FIT_CENTER);

        // 2. Cria o AlertDialog
        AlertDialog.Builder builder = new AlertDialog.Builder(getContext());
        builder.setView(dialogImageView); // Define o ImageView como o conteúdo do dialog

        // 3. Cria e exibe o dialog
        AlertDialog dialog = builder.create();

        // Adiciona um clique NA IMAGEM DENTRO DO DIALOG para fechá-lo
        dialogImageView.setOnClickListener(v -> dialog.dismiss());

        dialog.show();
    }


    @Override
    public void onDestroy() {
        super.onDestroy();
        if (interpreter != null) {
            interpreter.close();
        }
    }
}
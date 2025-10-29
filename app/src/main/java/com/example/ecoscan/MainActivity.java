package com.example.ecoscan;

import android.os.Bundle;
import android.view.MenuItem;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.navigation.NavigationBarView;

public class MainActivity extends AppCompatActivity {

    private BottomNavigationView bottomNavigationView;

    // Nossos 3 fragments
    private final Fragment scanFragment = new ScanFragment();
    private final Fragment infoFragment = new InfoFragment();
    private final Fragment settingsFragment = new SettingsFragment();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bottomNavigationView = findViewById(R.id.bottom_navigation);

        // Define o listener para cliques nos itens
        bottomNavigationView.setOnItemSelectedListener(new NavigationBarView.OnItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem item) {
                // Seleciona o fragment com base no item clicado
                if (item.getItemId() == R.id.nav_info) {
                    loadFragment(infoFragment);
                    return true;
                } else if (item.getItemId() == R.id.nav_scan) {
                    loadFragment(scanFragment);
                    return true;
                } else if (item.getItemId() == R.id.nav_settings) {
                    loadFragment(settingsFragment);
                    return true;
                }
                return false;
            }
        });

        // Carrega o fragment principal (ScanFragment) por padrão
        if (savedInstanceState == null) {
            bottomNavigationView.setSelectedItemId(R.id.nav_scan);
            loadFragment(scanFragment);
        }
    }

    // Método helper para carregar um fragment no container
    private void loadFragment(Fragment fragment) {
        FragmentManager fragmentManager = getSupportFragmentManager();
        FragmentTransaction fragmentTransaction = fragmentManager.beginTransaction();

        // Substitui o conteúdo do 'fragment_container' pelo novo fragment
        fragmentTransaction.replace(R.id.fragment_container, fragment);

        // Adiciona a transação à pilha de "voltar" (opcional, mas bom)
        // fragmentTransaction.addToBackStack(null);

        // Executa a transação
        fragmentTransaction.commit();
    }
}
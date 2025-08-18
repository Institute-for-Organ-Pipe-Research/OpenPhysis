# Dokumentacja Menadżera Organów

## 📌 Wprowadzenie

Do zarządzania organami wykorzystuje się managery wiatrownicy (divisions) i głosów (stops).

- `OrganDivisionManager` - zarządza wiatrownicami (np. Great, Swell, Pedal)
- `OrganStopManager` - zarządza głosami organowymi

```cpp
// Inicjalizacja menadżerów
OrganDivisionManager divisions;
OrganStopManager stops;
```

## Zarządzanie Wiatrownicami (Divisions)

### Tworzenie wiatrownicy

```c++
// Podstawowe wiatrownice
auto& great = divisions.addDivision("Great");  // Wiatrownica główna
auto& swell = divisions.addDivision("Swell"); // Wiatrownica ekspresyjna
auto& pedal = divisions.addDivision("Pedal"); // Wiatrownica pedałowa
```

### Operacje na wiatrownicach

```c++
// Pobieranie wiatrownicy
auto* division = divisions.getDivision(great.getDivisionID());

// Usuwanie wiatrownicy (uwaga: usuwa również wszystkie przypisane głosy!)
divisions.removeDivision(swell.getDivisionID());
```

## Zarządzanie Głosami (Stops)

### Tworzenie głosów

#### Głosy podstawowe

```c++
// Great Division
stops.addOrganStop(great.getDivisionID(), "Principal 8'",  OrganStop::Footage(8));
stops.addOrganStop(great.getDivisionID(), "Octave 4'",    OrganStop::Footage(4));
```

#### Głosy alikwotowe

Parametry konstruktora **Footage**:

- mainNumber - liczba całkowita (np. 2)
- numerator - licznik ułamka (np. 2)
- denominator - mianownik ułamka (np. 3)

```c++
// Przykłady głosów ułamkowych
stops.addOrganStop(great.getDivisionID(), "Nazard 2 2/3'", OrganStop::Footage(2, 2, 3));
stops.addOrganStop(great.getDivisionID(), "Tierce 1 3/5'", OrganStop::Footage(1, 3, 5));
```

### Sterowanie głosami

```c++
// Włączanie/wyłączanie głosu
stops.setStopEnabled(stopID, true);  // Włącz
stops.setStopEnabled(stopID, false); // Wyłącz

// Pobieranie informacji o głosie
if (OrganStop* stop = stops.getOrganStop(stopID)) {
    std::cout << "Nazwa: " << stop->getStopName() 
              << " Stopaż: " << stop->getFootage().toString();
}
```

## Przykład Kompletnej Konfiguracji

```c++
OrganDivisionManager divisions;
OrganStopManager stops;

// Konfiguracja wiatrownic
auto& great = divisions.addDivision("Great");
auto& swell = divisions.addDivision("Swell");

// Great Division
stops.addOrganStop(great.getDivisionID(), "Principal 8'",  OrganStop::Footage(8));
stops.addOrganStop(great.getDivisionID(), "Octave 4'",     OrganStop::Footage(4));
stops.addOrganStop(great.getDivisionID(), "Quint 2 2/3'",  OrganStop::Footage(2, 2, 3));

// Swell Division
stops.addOrganStop(swell.getDivisionID(), "Gedackt 8'",   OrganStop::Footage(8));
stops.addOrganStop(swell.getDivisionID(), "Flute 4'",     OrganStop::Footage(4));
```

### Ważne Uwagi

1. Kolejność działań

> Stwórz wiatrownicę -> Dodaj głosy -> Zarządzaj głosami

2. Bezpieczeństwo

```c++
// Zawsze sprawdzaj czy wiatrownica istnieje
if (auto* div = divisions.getDivision(id)) {
    // Operacje na division
}
```

3. Konwencje nazewnicze

    - Nazwy głosów: "Nazwa stopaż" (np. "Principal 8'")
    - Wiatrownice: Great, Swell, Pedal

4. Typowe stopaże

| Typ głosu  | Przykłady      |
|------------|----------------|
| Podstawowe | 16', 8', 4'    |
| Alikwoty   | 2 2/3', 1 3/5' |


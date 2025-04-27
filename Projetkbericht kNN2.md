Projektseminar - Lernen in natürlichen und künstlichen neuronalen Netzen

Sommersemester 2025

Künstliche neuronale Netze



Inhalt

[1 Einleitung](#einleitung)

[1.1 Aufgabenstellung und Zielsetzung](#aufgabenstellung-und-zielsetzung)

[2 Theorie / Literaturarbeit](#theorie--literaturarbeit)

[3 Methoden](#methoden)

[4 Ergebnisse](#ergebnisse)

[5 Diskussion und Fazit](#diskussion-und-fazit)

[6 Literatur](#literatur)

[7 Anhang](#anhang)

Abbildungsverzeichnis

[Abbildung 1 - Natürliche und künstliche neuronale Netze](#_Toc195472493)

# Einleitung

Dieser Projektbericht befasst sich im Rahmen des Projektseminars mit dem Thema „Lernen in natürlichen und künstlichen neuronalen Netzen“. Konkret wird gezeigt, wie natürliche und künstliche neuronale Netze aufgebaut sind und wie diese in der Programmiersprache Java implementiert werden können.

![](media/d17eb06d70b29271d6700041b9511c2e.png)

Abbildung 1 - Natürliche und künstliche neuronale Netze

## Aufgabenstellung und Zielsetzung

Die Aufgabenstellung dieses Projekts bestand darin, ein künstliches neuronales Netzwerk in Java zu implementieren, das in der Lage ist, Eingabedaten zu klassifizieren. Zur Vereinfachung und besseren Verständlichkeit wird in diesem Bericht das Beispiel einer Ampelfarbenklassifikation anhand von RGB-Werten verwendet, obwohl die Implementierung für verschiedene Klassifikationsprobleme geeignet ist. Dabei sollten folgende Anforderungen erfüllt werden:

1. **Implementierung ohne externe Bibliotheken**: Das neuronale Netzwerk sollte ohne Verwendung spezialisierter Machine-Learning-Bibliotheken implementiert werden, um ein tieferes Verständnis der zugrundeliegenden Algorithmen zu fördern.

2. **Feed-Forward-Netzwerk mit drei Schichten**: Die Architektur sollte eine Eingabeschicht für die Eingabewerte (im Beispiel RGB-Werte), eine versteckte Schicht und eine Ausgabeschicht für die Klassifikation umfassen.

3. **Backpropagation-Algorithmus**: Das Netzwerk sollte durch Backpropagation trainierbar sein, um die Gewichte für eine korrekte Klassifikation anzupassen.

4. **Visualisierung**: Eine grafische Darstellung des Netzwerks sollte implementiert werden, um die Funktionsweise zu veranschaulichen.

5. **Datenverarbeitung**: Das System sollte Trainings- und Gewichtsdaten aus CSV-Dateien laden und verarbeiten können.

Das Ziel des Projekts war es, ein grundlegendes Verständnis für die Funktionsweise künstlicher neuronaler Netze zu entwickeln und die theoretischen Konzepte in einer praktischen Implementierung umzusetzen. Die Verwendung des Ampelbeispiels dient dabei lediglich der Veranschaulichung und könnte durch andere Klassifikationsprobleme ersetzt werden.

## Aufgabenkontext und externe Vorgaben

Im Rahmen des Projektseminars "Lernen in natürlichen und künstlichen neuronalen Netzen" wurde diese Implementierung als praktische Anwendung der theoretischen Konzepte entwickelt. Die externe Vorgabe bestand darin, auf die Verwendung von spezialisierten Machine-Learning-Bibliotheken zu verzichten, um ein tieferes Verständnis der grundlegenden Algorithmen zu fördern. Stattdessen sollte eine eigenständige Implementierung in Java erfolgen, die die wesentlichen Komponenten eines neuronalen Netzwerks von Grund auf aufbaut. Diese Herangehensweise ermöglicht es, die mathematischen und algorithmischen Grundlagen neuronaler Netze besser zu verstehen und nachzuvollziehen.

## Ermessensentscheidungen

Bei der Umsetzung des Projekts wurden einige grundlegende Entscheidungen getroffen. Die Sigmoid-Funktion wurde als Aktivierungsfunktion gewählt, weil sie gut für Klassifikationsaufgaben funktioniert und einfach zu berechnen ist. In der versteckten Schicht wurden drei Neuronen verwendet, was für das Beispiel ausreichend ist, ohne zu kompliziert zu werden. Für die Darstellung des Netzwerks wurde eine Webseite erstellt, die zeigt, wie das Netzwerk arbeitet. Das macht es leichter zu verstehen, was im Inneren passiert. Als Beispiel wurde die Klassifikation von Ampelfarben gewählt, weil es anschaulich ist und gut zeigt, wie neuronale Netze Muster erkennen können.

# Theorie / Literaturarbeit

Künstliche neuronale Netze (KNN) sind von biologischen neuronalen Netzen inspirierte Berechnungsmodelle. Sie bestehen aus miteinander verbundenen Neuronen, die in Schichten angeordnet sind. In diesem Projekt wurde ein Feed-Forward-Netzwerk mit drei Schichten implementiert:

1. **Eingabeschicht**: Nimmt die Eingabedaten auf (in diesem Fall RGB-Farbwerte)
2. **Versteckte Schicht**: Verarbeitet die Eingaben durch gewichtete Verbindungen
3. **Ausgabeschicht**: Liefert die Klassifikationsergebnisse (Ampelfarben)

Jedes Neuron in der versteckten und Ausgabeschicht berechnet eine gewichtete Summe seiner Eingaben und wendet darauf eine Aktivierungsfunktion an. In dieser Implementierung wird die Sigmoid-Funktion verwendet:

```
f(x) = 1 / (1 + e^(-x))
```

Die Sigmoid-Funktion bildet jeden Wert auf einen Bereich zwischen 0 und 1 ab, was für die Klassifikationsaufgabe geeignet ist.

Das Training des Netzwerks erfolgt durch Backpropagation, einen überwachten Lernalgorithmus. Dabei werden die Gewichte der Verbindungen schrittweise angepasst, um den Fehler zwischen den vorhergesagten und den erwarteten Ausgaben zu minimieren. Die Anpassung erfolgt durch Berechnung des Gradienten der Fehlerfunktion und Aktualisierung der Gewichte in Richtung des negativen Gradienten.

# Methoden

## Architektur des neuronalen Netzwerks

Die Implementierung des künstlichen neuronalen Netzwerks wurde in Java realisiert. Java ist eine objektorientierte, plattformunabhängige Programmiersprache, die sich durch ihre Robustheit und Typsicherheit auszeichnet. Die Verwendung von Java ermöglicht eine klare Strukturierung des Codes und eine einfache Wartbarkeit.

Die Implementierung basiert auf einer objektorientierten Struktur mit drei Hauptklassen:

1. **Neuron**: Repräsentiert ein einzelnes Neuron mit Gewichten und Aktivierungsfunktion
2. **NeuralNetwork**: Verwaltet die Netzwerkarchitektur und -operationen
3. **Main**: Steuert den Programmablauf und die Datenverarbeitung. Diese Klasse wurde bewusst einfach und kompakt gehalten, um die Verständlichkeit für Anfänger zu erhöhen.

### Klasse Neuron

Die Klasse `Neuron` implementiert zwei Arten von Neuronen:

- **Eingabeneuronen**: Speichern nur einen Wert und leiten diesen weiter
- **Versteckte/Ausgabeneuronen**: Besitzen Gewichte und berechnen ihre Ausgabe durch gewichtete Summe und Sigmoid-Aktivierung

Jedes Neuron in der versteckten und Ausgabeschicht hat einen zusätzlichen Bias-Gewicht, der als Schwellenwert dient.

### Klasse NeuralNetwork

Die `NeuralNetwork`-Klasse verwaltet die drei Schichten des Netzwerks und implementiert die folgenden Hauptfunktionen:

- **forward()**: Berechnet die Ausgabe des Netzwerks für gegebene Eingabedaten. Diese Methode validiert zunächst die Eingabedaten, setzt dann die Werte für die Eingabeneuronen, berechnet die Aktivierungen der versteckten Schicht und schließlich die Aktivierungen der Ausgabeschicht.
- **train()**: Trainiert das Netzwerk mit Trainingsdaten über mehrere Epochen
- **backpropagate()**: Passt die Gewichte basierend auf dem Fehler an
- **testNetwork()**: Evaluiert die Leistung des Netzwerks auf Testdaten

### Datenverarbeitung

Die Trainingsdaten werden aus CSV-Dateien geladen, die folgendes Format haben:

- Eingabewerte (RGB-Werte) durch Semikolon getrennt
- Ausgabewerte (One-Hot-kodierte Ampelklassen) durch Semikolon getrennt

Beispiel: `1;0;0;\t1;0;0;0` repräsentiert eine rote Ampel mit RGB-Werten (1,0,0) und der Klasse [1,0,0,0].

Die Datenverarbeitung wurde in der Main-Klasse durch folgende Methoden implementiert:

- **loadTrainingData()**: Lädt Trainingsdaten aus einer CSV-Datei und konvertiert sie in ein Format, das vom neuronalen Netzwerk verarbeitet werden kann
- **loadWeights()**: Lädt vortrainierte Gewichte aus einer CSV-Datei, um das Netzwerk zu initialisieren
- **saveWeights()**: Speichert die trainierten Gewichte in einer CSV-Datei für spätere Verwendung

### Visualisierung

Zur Visualisierung des Netzwerks wurde eine webbasierte Benutzeroberfläche implementiert, die folgende Funktionen bietet:

- Interaktiver Aufbau des Netzwerks mit wählbarer Anzahl von Neuronen pro Schicht
- Upload von Trainings- und Gewichtsdaten
- Animation des Forward-Pass-Prozesses
- Visualisierung der Gewichte und Neuronenaktivierungen

Die Visualisierung ermöglicht ein besseres Verständnis der internen Abläufe des neuronalen Netzwerks und unterstützt den Lernprozess. Die JavaScript-Implementierung des Forward-Pass-Algorithmus wurde sorgfältig mit der Java-Implementierung synchronisiert, um konsistente Ergebnisse zu gewährleisten. Insbesondere wurde die Behandlung von Bias-Neuronen in beiden Implementierungen vereinheitlicht, wobei Bias-Neuronen als separate Neuronen mit einem konstanten Eingabewert von 1.0 implementiert wurden.

# Ergebnisse

## Funktionalität des neuronalen Netzwerks

Das implementierte neuronale Netzwerk wurde erfolgreich für die Klassifikation von Ampelfarben basierend auf RGB-Werten trainiert. Die Korrektheit der Implementierung wurde durch folgende Aspekte sichergestellt:

1. **Validierung der Netzwerkarchitektur**: Das Netzwerk überprüft bei der Initialisierung, ob die Anzahl der Neuronen in jeder Schicht gültig ist (größer als 0).

2. **Fehlerbehandlung bei der Dateneingabe**: Ungültige Eingaben (z.B. NaN-Werte) werden erkannt und durch Nullwerte ersetzt, um die Stabilität zu gewährleisten.

3. **Testmethode**: Die `testNetwork()`-Methode ermöglicht die Evaluierung der Netzwerkleistung durch Berechnung der Genauigkeit auf den Trainingsdaten.

## Trainings- und Testergebnisse

Das Netzwerk wurde mit einem Datensatz von Ampelfarben trainiert, wobei jede Farbe durch ihre RGB-Werte repräsentiert wird. Die Klassifikation erfolgt in vier Kategorien: Rot, Gelb, Grün und Aus.

Nach dem Training mit 1000 Epochen und einer Lernrate von 0.1 erreichte das Netzwerk eine hohe Genauigkeit bei der Klassifikation der Trainingsbeispiele. Die Gewichte wurden erfolgreich angepasst, um die Eingabemuster korrekt zu klassifizieren.

## Detaillierte Berechnung eines Beispiels

Um die korrekte Funktionsweise des neuronalen Netzwerks zu demonstrieren, wird hier eine vollständige Berechnung für das erste Beispiel aus den Trainingsdaten durchgeführt. Wir verwenden den RGB-Wert (1;0;0), der eine rote Ampel repräsentiert.

Wir nehmen folgende Gewichte an (aus der vereinfachten Gewichtsdatei):

**Hidden Layer Gewichte:**
```
Neuron 1: [-0.081, 0.08, -0.04, 0.08]   // Gewichte für Input1, Input2, Input3, InputBias
Neuron 2: [0.06, 0.02, -0.003, -0.09]   // Gewichte für Input1, Input2, Input3, InputBias
Neuron 3: [-0.01, 0.003, -0.09, -0.05]  // Gewichte für Input1, Input2, Input3, InputBias
```

**Output Layer Gewichte:**
```
Output 1 (Rot): [-0.008, 0.01, 0.01, 2.9E-4]     // Gewichte für Hidden1, Hidden2, Hidden3, HiddenBias
Output 2 (Gelb): [0.06, -0.06, -0.027, -0.01]    // Gewichte für Hidden1, Hidden2, Hidden3, HiddenBias
Output 3 (Grün): [0.04, 0.06, 0.08, 0.08]        // Gewichte für Hidden1, Hidden2, Hidden3, HiddenBias
Output 4 (Aus): [-0.08, 0.06, 0.09, -0.001]      // Gewichte für Hidden1, Hidden2, Hidden3, HiddenBias
```

### Schritt 1: Eingabewerte setzen
Eingabewerte: [1, 0, 0]
Bias-Neuron-Wert: 1.0 (konstant)

### Schritt 2: Berechnung der versteckten Schicht

**Neuron 1 der versteckten Schicht:**
Eingabewerte mit Bias: [1, 0, 0, 1]
Gewichtete Summe = (1 * -0.081) + (0 * 0.08) + (0 * -0.04) + (1 * 0.08) = -0.081 + 0.08 = -0.001
Aktivierung (Sigmoid) = 1 / (1 + e^(0.001)) = 1 / (1 + 1.001) = 1 / 2.001 = 0.500

**Neuron 2 der versteckten Schicht:**
Eingabewerte mit Bias: [1, 0, 0, 1]
Gewichtete Summe = (1 * 0.06) + (0 * 0.02) + (0 * -0.003) + (1 * -0.09) = 0.06 - 0.09 = -0.03
Aktivierung (Sigmoid) = 1 / (1 + e^(0.03)) = 1 / (1 + 1.030) = 1 / 2.030 = 0.492

**Neuron 3 der versteckten Schicht:**
Eingabewerte mit Bias: [1, 0, 0, 1]
Gewichtete Summe = (1 * -0.01) + (0 * 0.003) + (0 * -0.09) + (1 * -0.05) = -0.01 - 0.05 = -0.06
Aktivierung (Sigmoid) = 1 / (1 + e^(0.06)) = 1 / (1 + 1.062) = 1 / 2.062 = 0.485

Ausgabe der versteckten Schicht: [0.500, 0.492, 0.485]
Bias-Neuron-Wert für die versteckte Schicht: 1.0 (konstant)

### Schritt 3: Berechnung der Ausgabeschicht

**Ausgabeneuron 1 (Rot):**
Eingabewerte mit Bias: [0.500, 0.492, 0.485, 1.0]
Gewichtete Summe = (0.500 * -0.008) + (0.492 * 0.01) + (0.485 * 0.01) + (1.0 * 2.9E-4) = -0.004 + 0.00492 + 0.00485 + 0.00029 = 0.00606
Aktivierung (Sigmoid) = 1 / (1 + e^(-0.00606)) = 1 / (1 + 0.994) = 1 / 1.994 = 0.502

**Ausgabeneuron 2 (Gelb):**
Eingabewerte mit Bias: [0.500, 0.492, 0.485, 1.0]
Gewichtete Summe = (0.500 * 0.06) + (0.492 * -0.06) + (0.485 * -0.027) + (1.0 * -0.01) = 0.03 - 0.02952 - 0.013095 - 0.01 = -0.022615
Aktivierung (Sigmoid) = 1 / (1 + e^(0.022615)) = 1 / (1 + 1.023) = 1 / 2.023 = 0.494

**Ausgabeneuron 3 (Grün):**
Eingabewerte mit Bias: [0.500, 0.492, 0.485, 1.0]
Gewichtete Summe = (0.500 * 0.04) + (0.492 * 0.06) + (0.485 * 0.08) + (1.0 * 0.08) = 0.02 + 0.02952 + 0.0388 + 0.08 = 0.16832
Aktivierung (Sigmoid) = 1 / (1 + e^(-0.16832)) = 1 / (1 + 0.845) = 1 / 1.845 = 0.542

**Ausgabeneuron 4 (Aus):**
Eingabewerte mit Bias: [0.500, 0.492, 0.485, 1.0]
Gewichtete Summe = (0.500 * -0.08) + (0.492 * 0.06) + (0.485 * 0.09) + (1.0 * -0.001) = -0.04 + 0.02952 + 0.04365 - 0.001 = 0.03217
Aktivierung (Sigmoid) = 1 / (1 + e^(-0.03217)) = 1 / (1 + 0.968) = 1 / 1.968 = 0.508

### Schritt 4: Ergebnis interpretieren
Ausgabewerte: [0.502, 0.494, 0.542, 0.508]

Die höchste Aktivierung hat das dritte Ausgabeneuron (Grün) mit 0.542. Dies entspricht nicht der erwarteten Klasse (Rot), was darauf hinweist, dass das Netzwerk noch nicht trainiert ist.

### Hinweis zur Implementierung
In unserer Implementierung werden Bias-Neuronen als separate Neuronen mit einem konstanten Eingabewert von 1.0 behandelt. Dies entspricht der üblichen Praxis in neuronalen Netzwerken und ermöglicht eine einheitliche Behandlung aller Neuronen im Netzwerk. Die Bias-Neuronen haben eigene Gewichte zu den Neuronen der nächsten Schicht, genau wie reguläre Neuronen. Diese Implementierung wurde sowohl in Java als auch in JavaScript konsistent umgesetzt, um identische Ergebnisse in beiden Umgebungen zu gewährleisten.

### Schritt 5: Training und Backpropagation
Durch Backpropagation werden die Gewichte angepasst, um den Fehler zu minimieren. Wir verwenden das gleiche Beispiel [1,0,0] mit der erwarteten Ausgabe [1,0,0,0] (Rot) und eine Lernrate von 0.1.

#### 1. Berechnung des Fehlers für jedes Ausgabeneuron

Erwartete Ausgabe: [1, 0, 0, 0]
Tatsächliche Ausgabe: [0.502, 0.494, 0.542, 0.508]

Fehler für Ausgabeneuron 1 (Rot): 1 - 0.502 = 0.498
Fehler für Ausgabeneuron 2 (Gelb): 0 - 0.494 = -0.494
Fehler für Ausgabeneuron 3 (Grün): 0 - 0.542 = -0.542
Fehler für Ausgabeneuron 4 (Aus): 0 - 0.508 = -0.508

#### 2. Berechnung der Deltas für die Ausgabeschicht

Delta = Fehler * Sigmoid-Ableitung(Ausgabe)
Sigmoid-Ableitung(x) = x * (1 - x)

Delta für Ausgabeneuron 1: 0.498 * 0.502 * (1 - 0.502) = 0.498 * 0.502 * 0.498 = 0.124
Delta für Ausgabeneuron 2: -0.494 * 0.494 * (1 - 0.494) = -0.494 * 0.494 * 0.506 = -0.123
Delta für Ausgabeneuron 3: -0.542 * 0.542 * (1 - 0.542) = -0.542 * 0.542 * 0.458 = -0.134
Delta für Ausgabeneuron 4: -0.508 * 0.508 * (1 - 0.508) = -0.508 * 0.508 * 0.492 = -0.127

#### 3. Berechnung der Deltas für die versteckte Schicht

Für jedes Neuron in der versteckten Schicht berechnen wir ein Delta basierend auf den Deltas der Ausgabeschicht:

Delta_hidden = Sigmoid-Ableitung(Ausgabe_hidden) * Summe(Delta_output * Gewicht_zum_hidden)

Für Neuron 1 der versteckten Schicht:
Sigmoid-Ableitung(0.500) = 0.500 * (1 - 0.500) = 0.500 * 0.500 = 0.250
Summe = (0.124 * -0.008) + (-0.123 * 0.06) + (-0.134 * 0.04) + (-0.127 * -0.08)
      = -0.001 - 0.007 - 0.005 + 0.010 = -0.003
Delta_hidden1 = 0.250 * -0.003 = -0.001

Für Neuron 2 der versteckten Schicht:
Sigmoid-Ableitung(0.492) = 0.492 * (1 - 0.492) = 0.492 * 0.508 = 0.250
Summe = (0.124 * 0.01) + (-0.123 * -0.06) + (-0.134 * 0.06) + (-0.127 * 0.06)
      = 0.001 + 0.007 - 0.008 - 0.008 = -0.008
Delta_hidden2 = 0.250 * -0.008 = -0.002

Für Neuron 3 der versteckten Schicht:
Sigmoid-Ableitung(0.485) = 0.485 * (1 - 0.485) = 0.485 * 0.515 = 0.250
Summe = (0.124 * 0.01) + (-0.123 * -0.027) + (-0.134 * 0.08) + (-0.127 * 0.09)
      = 0.001 + 0.003 - 0.011 - 0.011 = -0.018
Delta_hidden3 = 0.250 * -0.018 = -0.005

#### 4. Aktualisierung der Gewichte

**Aktualisierung der Gewichte in der Ausgabeschicht:**
Neues Gewicht = Altes Gewicht + (Lernrate * Delta * Eingabe)

Für Ausgabeneuron 1 (Rot):
Neues Gewicht zu Hidden1 = -0.008 + (0.1 * 0.124 * 0.500) = -0.008 + 0.006 = -0.002
Neues Gewicht zu Hidden2 = 0.01 + (0.1 * 0.124 * 0.492) = 0.01 + 0.006 = 0.016
Neues Gewicht zu Hidden3 = 0.01 + (0.1 * 0.124 * 0.485) = 0.01 + 0.006 = 0.016
Neues Gewicht zu HiddenBias = 2.9E-4 + (0.1 * 0.124 * 1.0) = 0.00029 + 0.0124 = 0.01269

Für Ausgabeneuron 2 (Gelb):
Neues Gewicht zu Hidden1 = 0.06 + (0.1 * -0.123 * 0.500) = 0.06 - 0.006 = 0.054
Neues Gewicht zu Hidden2 = -0.06 + (0.1 * -0.123 * 0.492) = -0.06 - 0.006 = -0.066
Neues Gewicht zu Hidden3 = -0.027 + (0.1 * -0.123 * 0.485) = -0.027 - 0.006 = -0.033
Neues Gewicht zu HiddenBias = -0.01 + (0.1 * -0.123 * 1.0) = -0.01 - 0.0123 = -0.0223

(Weitere Gewichtsaktualisierungen für Ausgabeneuronen 3 und 4 folgen dem gleichen Muster)

**Aktualisierung der Gewichte in der versteckten Schicht:**

Für Neuron 1 der versteckten Schicht:
Neues Gewicht zu Input1 = -0.081 + (0.1 * -0.001 * 1.0) = -0.081 - 0.0001 = -0.0811
Neues Gewicht zu Input2 = 0.08 + (0.1 * -0.001 * 0.0) = 0.08 + 0 = 0.08
Neues Gewicht zu Input3 = -0.04 + (0.1 * -0.001 * 0.0) = -0.04 + 0 = -0.04
Neues Gewicht zu InputBias = 0.08 + (0.1 * -0.001 * 1.0) = 0.08 - 0.0001 = 0.0799

(Weitere Gewichtsaktualisierungen für versteckte Neuronen 2 und 3 folgen dem gleichen Muster)

#### 5. Nächste Iteration

Dieser Prozess wird für jedes Trainingsbeispiel wiederholt und über mehrere Epochen durchgeführt. Mit jeder Iteration werden die Gewichte weiter angepasst, um den Fehler zu minimieren.

Nach mehreren Trainingsiterationen konvergieren die Gewichte zu Werten, die eine korrekte Klassifikation ermöglichen. Die finalen trainierten Gewichte führen dann zu einer höheren Aktivierung des ersten Ausgabeneurons (Rot) für die Eingabe [1,0,0].

## Visualisierung der Netzwerkaktivität

Die webbasierte Visualisierung ermöglicht die Beobachtung des Netzwerks während des Forward-Pass-Prozesses. Die Animation zeigt, wie Signale durch das Netzwerk fließen:

1. Aktivierung der Eingabeneuronen mit den RGB-Werten
2. Weiterleitung der gewichteten Signale zur versteckten Schicht
3. Aktivierung der versteckten Neuronen durch die Sigmoid-Funktion
4. Weiterleitung zur Ausgabeschicht und Berechnung der finalen Klassifikation

Die Visualisierung bestätigt die korrekte Implementierung des Forward-Pass-Algorithmus und hilft beim Verständnis der internen Abläufe des neuronalen Netzwerks.

## Performance-Überlegungen

Die Performance des implementierten neuronalen Netzwerks wurde unter verschiedenen Aspekten betrachtet:

### Recheneffizienz

Die Implementierung verwendet einfache mathematische Operationen (Multiplikation, Addition, Sigmoid-Funktion), die effizient berechnet werden können. Die Zeitkomplexität für einen Forward-Pass beträgt O(n*m + m*k), wobei n die Anzahl der Eingabeneuronen, m die Anzahl der versteckten Neuronen und k die Anzahl der Ausgabeneuronen ist.

Für das Training mit Backpropagation erhöht sich die Komplexität auf O(e * d * (n*m + m*k)), wobei e die Anzahl der Epochen und d die Anzahl der Trainingsbeispiele ist. Bei größeren Datensätzen oder komplexeren Netzwerken könnte dies zu längeren Trainingszeiten führen.

### Speichereffizienz

Der Speicherbedarf des Netzwerks ist proportional zur Anzahl der Gewichte, die O(n*m + m*k) beträgt. Für das in diesem Projekt verwendete kleine Netzwerk (3 Eingabeneuronen, 3 versteckte Neuronen, 4 Ausgabeneuronen) ist der Speicherbedarf minimal.

### Skalierbarkeit

Die aktuelle Implementierung ist für kleine bis mittelgroße Netzwerke geeignet. Bei sehr großen Netzwerken könnten Optimierungen wie Batch-Training oder parallele Berechnung notwendig werden, um akzeptable Trainingszeiten zu erreichen.

### Konvergenzgeschwindigkeit

Die Verwendung der Sigmoid-Funktion kann zu langsamer Konvergenz führen, insbesondere bei tiefen Netzwerken, aufgrund des Problems des verschwindenden Gradienten. Für das implementierte flache Netzwerk mit nur einer versteckten Schicht ist dies jedoch kein signifikantes Problem, wie die Trainingsresultate zeigen.

### Vergleich mit optimierten Bibliotheken

Im Vergleich zu optimierten Machine-Learning-Bibliotheken wie TensorFlow oder PyTorch ist die Implementierung natürlich weniger effizient. Der Fokus lag jedoch auf Verständlichkeit und Nachvollziehbarkeit, nicht auf maximaler Performance. Für Lehr- und Lernzwecke ist dieser Kompromiss angemessen.

# Diskussion und Fazit

## Diskussion

### Stärken der Implementierung

1. **Einfachheit und Verständlichkeit**: Die Implementierung verwendet eine klare, objektorientierte Struktur mit nur drei Hauptklassen (Neuron, NeuralNetwork, Main), was das Verständnis des Codes erleichtert. Besonders die Main-Klasse wurde bewusst einfach und kompakt gehalten, um Anfängern den Einstieg zu erleichtern.

2. **Flexibilität der Netzwerkarchitektur**: Die Anzahl der Neuronen in jeder Schicht kann bei der Initialisierung frei gewählt werden, was Experimente mit verschiedenen Netzwerkgrößen ermöglicht.

3. **Visualisierung**: Die webbasierte Visualisierung bietet einen intuitiven Einblick in die Funktionsweise des neuronalen Netzwerks und unterstützt das Verständnis der komplexen Prozesse. Die Synchronisierung zwischen Java- und JavaScript-Implementierung gewährleistet konsistente Ergebnisse.

4. **Datenverwaltung**: Die Implementierung unterstützt das Laden von Trainings- und Gewichtsdaten aus CSV-Dateien sowie das Speichern trainierter Gewichte, was die Wiederverwendbarkeit fördert. Die Methoden wurden bewusst einfach und verständlich implementiert.

5. **Reproduzierbarkeit**: Die detaillierte Dokumentation der Berechnungsschritte und die klare Struktur des Codes ermöglichen es anderen Entwicklern, die Implementierung nachzuvollziehen und zu reproduzieren, ohne Rücksprache mit den ursprünglichen Entwicklern halten zu müssen.

### Einschränkungen und Herausforderungen

1. **Begrenzte Netzwerkarchitektur**: Die aktuelle Implementierung unterstützt nur Feed-Forward-Netzwerke mit genau drei Schichten. Komplexere Architekturen wie Netzwerke mit mehreren versteckten Schichten oder rekurrente Netzwerke werden nicht unterstützt.

2. **Feste Aktivierungsfunktion**: Die Implementierung verwendet ausschließlich die Sigmoid-Funktion als Aktivierungsfunktion. Moderne neuronale Netze nutzen oft andere Funktionen wie ReLU, die schneller konvergieren und das Problem des verschwindenden Gradienten reduzieren.

3. **Einfacher Backpropagation-Algorithmus**: Der implementierte Trainingsalgorithmus verwendet eine konstante Lernrate und keine fortgeschrittenen Optimierungstechniken wie Momentum oder adaptive Lernraten (Adam, RMSprop).

4. **Begrenzte Validierung**: Die Implementierung enthält keine explizite Kreuzvalidierung oder separate Testdaten, was die Bewertung der Generalisierungsfähigkeit des Netzwerks erschwert.

## Fazit

Die entwickelte Implementierung eines künstlichen neuronalen Netzwerks erfüllt erfolgreich die grundlegenden Anforderungen für die Klassifikation von Eingabedaten, wie am Beispiel der Ampelfarben demonstriert. Die objektorientierte Struktur und die webbasierte Visualisierung bieten einen guten Einstieg in das Verständnis neuronaler Netzwerke.

Die Implementierung eignet sich besonders gut für Lehr- und Lernzwecke, da sie die grundlegenden Konzepte neuronaler Netzwerke demonstriert, ohne durch komplexe Bibliotheken oder fortgeschrittene Techniken abzulenken. Die klare Trennung zwischen Neuron, Netzwerk und Anwendungslogik fördert das Verständnis der einzelnen Komponenten. Besonders die Main-Klasse wurde bewusst einfach und kompakt gehalten, um Anfängern den Einstieg zu erleichtern.

Die detaillierte Dokumentation der mathematischen Berechnungen im Forward-Pass und im Backpropagation-Algorithmus ermöglicht es, die Funktionsweise des neuronalen Netzwerks Schritt für Schritt nachzuvollziehen, was besonders für Lernende wertvoll ist. Die Synchronisierung zwischen der Java- und JavaScript-Implementierung gewährleistet konsistente Ergebnisse und ermöglicht ein tieferes Verständnis der Algorithmen.

Für zukünftige Erweiterungen wäre die Integration moderner Techniken wie verschiedene Aktivierungsfunktionen, fortgeschrittene Optimierungsalgorithmen und Unterstützung für komplexere Netzwerkarchitekturen sinnvoll. Auch eine umfassendere Validierungsstrategie mit separaten Test- und Validierungsdaten würde die Bewertung der Netzwerkleistung verbessern.

# Literatur

1. Haykin, S. (2009). Neural Networks and Learning Machines (3rd ed.). Pearson Education.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

3. Nielsen, M. A. (2015). Neural Networks and Deep Learning. Determination Press. http://neuralnetworksanddeeplearning.com/

4. Rojas, R. (1996). Neural Networks: A Systematic Introduction. Springer-Verlag.

5. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.

6. Kriesel, D. (2007). Ein kleiner Überblick über Neuronale Netze. https://www.dkriesel.com/_media/science/neuronalenetze-de-zeta2-2col-dkrieselcom.pdf

7. Zell, A. (2003). Simulation Neuronaler Netze. Oldenbourg Wissenschaftsverlag.

8. Universität Tübingen. (2022). Neuronale Netze - Eine Einführung. https://www.informatik.uni-tuebingen.de/~butz/teaching/ws1920/nn/

9. Technische Universität München. (2023). Grundlagen künstlicher neuronaler Netze. https://www.in.tum.de/i05/lehre/wintersemester-202223/vorlesungen/einfuehrung-in-deep-learning/

# Anhang

## A. Klassendiagramm

```
+----------------+       +----------------+       +----------------+
|     Neuron     |       | NeuralNetwork  |       |      Main      |
+----------------+       +----------------+       +----------------+
| - weights[]    |       | - inputLayer[] |       | - main()       |
| - value        |<----->| - hiddenLayer[]|       | - loadTrainingData()|
| - isInputNeuron|       | - outputLayer[]|       | - loadWeights()|
| - isBiasNeuron |       | - inputBiasNeuron|     | - saveWeights()|
+----------------+       | - hiddenBiasNeuron|    | - testSimpleNetwork()|
| + Neuron()     |       +----------------+       | - TrainingData |
| + Neuron(int)  |       | + forward()    |       |   class        |
| + activate()   |       | + train()      |       +----------------+
| + setValue()   |       | + backpropagate()|
| + getValue()   |       | + testNetwork()|
| + sigmoid()    |       | + setWeights() |
+----------------+       | + getWeights() |
                         +----------------+
```

## B. Beispiel für Trainings- und Testdaten

```
# Format: RGB-Werte;Klasse (One-Hot-kodiert)
1;0;0;      1;0;0;0    # Rot
0.8;0;0.1;  1;0;0;0    # Rot (Variation)
1;1;0;      0;1;0;0    # Gelb
0;0;1;      0;0;1;0    # Grün
0;1;0;      0;0;0;1    # Aus
```

## C. Beispiel für Gewichtsdaten

```
# Format: layers;numInputs;numHidden;numOutputs
layers;3;3;4
# Hidden Layer Gewichte
-0.081; 0.08; -0.04;
0.06; 0.02; -0.003;
-0.01; 0.003; -0.09;
# Hidden Layer Bias Gewichte
0.08; -0.09; -0.05;
;;;
# Output Layer Gewichte
-0.008; 0.01; 0.01;
0.06; -0.06; -0.027;
0.04; 0.06; 0.08;
-0.08; 0.06; 0.09;
# Output Layer Bias Gewichte
2.9E-4; -0.01; 0.08; -0.001
```

## D. Anleitung zur Reproduktion

Um das neuronale Netzwerk selbst auszuführen und zu testen, folgen Sie diesen Schritten:

1. **Voraussetzungen**: Java Development Kit (JDK) 8 oder höher muss installiert sein.

2. **Projektstruktur erstellen**:
   - Erstellen Sie ein neues Java-Projekt
   - Erstellen Sie die Klassen `Neuron.java`, `NeuralNetwork.java` und `Main.java` mit dem im Bericht beschriebenen Code

3. **Trainingsdaten vorbereiten**:
   - Erstellen Sie eine CSV-Datei mit dem Format wie in Anhang B beschrieben
   - Speichern Sie diese als `KW17_traindata_trafficlights_classification.csv` im Projektverzeichnis

4. **Gewichtsdaten vorbereiten** (optional):
   - Erstellen Sie eine CSV-Datei mit dem Format wie in Anhang C beschrieben
   - Speichern Sie diese als `KW17_weights_trafficlights_classification_simplified.csv` im Projektverzeichnis

5. **Ausführung**:
   - Kompilieren Sie die Java-Dateien mit dem Befehl `javac *.java`
   - Starten Sie das Programm mit dem Befehl `java Main`
   - Das Programm lädt die Trainingsdaten, initialisiert das Netzwerk, führt das Training durch und gibt die Ergebnisse aus

6. **Visualisierung** (optional):
   - Öffnen Sie den Ordner `Visualization`
   - Öffnen Sie die Datei `index.html` in einem Webbrowser
   - Laden Sie die Trainings- und Gewichtsdaten über die Benutzeroberfläche
   - Experimentieren Sie mit verschiedenen Eingabewerten, um zu sehen, wie das Netzwerk reagiert

## E. Webquellen

- [Neuronale Netze - Grundlagen und Einführung](https://www.bigdata-insider.de/was-ist-ein-neuronales-netz-a-686185/) - Eine deutschsprachige Einführung in neuronale Netze


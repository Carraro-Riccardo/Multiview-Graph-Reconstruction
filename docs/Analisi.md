# Analisi della Pipeline di Ricostruzione di Grafi di Impianti Idraulici

## Panoramica Generale

Il sistema ricostruisce la topologia di un impianto idraulico a partire da sequenze di immagini multi-vista di singoli nodi dell'impianto. L'obiettivo finale è produrre un **grafo aciclico connesso** (albero) i cui nodi rappresentano componenti idraulici fisici (valvole, giunti, tee, croci, ecc.) e i cui archi rappresentano connessioni fisiche tra di essi.

L'impianto reale ha una struttura rigida e vincolata: ogni componente ha una **valenza fisica** (numero massimo di connessioni) determinata dalla sua tipologia. Il sistema sfrutta questa conoscenza a priori come vincolo duro durante la costruzione del grafo.

La pipeline è strutturata in due fasi principali:
- **Step 1**: Rilevazione e decodifica dei componenti da immagini
- **Step 4**: Ricostruzione del grafo tramite fusione multi-vista, scoring e ottimizzazione ILP

---

## Step 1 — Rilevazione e Decodifica

### Idea

Ogni nodo dell'impianto viene fotografato da più angolazioni. Ogni immagine è elaborata indipendentemente per rilevare i componenti visibili e identificarne il tipo tramite un codice visivo (Shurikode) stampato su ciascun componente.

### Processo

**1. Raggruppamento delle immagini per nodo** (`discover_nodes`)

Le immagini sono nominate secondo il pattern `node<id>_<frame>.jpg`. Il sistema raggruppa automaticamente tutte le immagini appartenenti allo stesso nodo fisico.

**2. Rilevazione con YOLO** (`detect_and_decode`)

Su ogni immagine viene applicato un modello YOLO fine-tuned per rilevare le bounding box dei componenti idraulici. I parametri sono volutamente conservativi (`YOLO_CONF=0.05`, `YOLO_IOU=0.4`, `YOLO_IMGSZ=1280`) per massimizzare il recall: è preferibile avere falsi positivi (filtrabili in seguito) piuttosto che perdere componenti reali.

Prima della rilevazione, l'immagine viene preprocessata con **CLAHE** (Contrast Limited Adaptive Histogram Equalization) per migliorare il contrasto locale e rendere i codici visivi più leggibili in condizioni di illuminazione non uniforme.

**3. Decodifica con Shurikode** (`decode_patch`)

Per ogni bounding box rilevata, il patch corrispondente viene estratto dall'immagine originale e passato a un decoder basato su **ResNet-50** che identifica il codice numerico del componente (es. 15 = Saracinesca, 116 = Tee FL). Il decoder restituisce anche una **confidenza** della predizione.

Le detections con confidenza inferiore a `CONF_THRESHOLD=0.5` vengono scartate (`"accepted": false`).

**4. Output per immagine**

Ogni rilevazione accettata produce un oggetto con:
- Bounding box normalizzata `[x1, y1, x2, y2]`
- Codice del componente e nome leggibile
- Centro normalizzato `(cx, cy)`
- Valenza fisica massima (`max_degree`)
- Flag `is_diameter` (per marker di diametro, non topologici)
- Confidenza del decoder

I risultati sono salvati in cache JSON per evitare ricalcoli.

---

## Step 4 — Ricostruzione del Grafo

### Idea Generale

Dato un insieme di detections per più viste dello stesso nodo, il sistema deve determinare **quali coppie di componenti sono fisicamente connesse**. Il problema è non banale perché:
- Le posizioni dei componenti nell'immagine variano con la prospettiva
- Un singolo componente può essere visibile in più viste con posizioni diverse
- La topologia reale è aciclica (albero)
- Ogni componente ha un numero fisso di porte di connessione

La soluzione adottata combina:
1. **Geometria** come filtro locale (i componenti vicini hanno più probabilità di essere connessi)
2. **Consenso multi-vista** come segnale semantico primario (se due tipi di componenti appaiono adiacenti in molte viste, è probabile che siano connessi)
3. **Priors statistici** appresi dal dataset (co-occorrenze, pattern a tripletta)
4. **Ottimizzazione globale ILP** per soddisfare simultaneamente tutti i vincoli fisici e produrre un albero

---

### Step 4.1 — Selezione della Vista Migliore

**Idea**: Non tutte le viste hanno la stessa qualità. La vista con più componenti topologici ben rilevati viene eletta come **vista di riferimento** per le posizioni 2D dei nodi nel grafo.

**Scoring**: `score = n_componenti_topologici + media_confidenze`

La vista migliore fornisce il sistema di coordinate spaziale. Le altre viste contribuiscono solo al consenso sui tipi di connessione, non alle posizioni.

---

### Step 4.2 — Consolidamento dei Codici nella Vista Migliore

**Idea**: Una detection nella vista migliore potrebbe avere confidenza bassa o essere ambigua. Se le altre viste concordano su un tipo diverso per una posizione simile, vale la pena considerare una sostituzione del codice.

**Processo** (`consolidate_best_view_codes`):
- Per ogni detection nella vista migliore, si valuta se esiste un'ipotesi alternativa di tipo meglio supportata dalle altre viste
- Il punteggio dell'ipotesi considera: supporto nelle altre viste, coerenza con i vicini locali, accordo sul tipo, compatibilità con i pattern a tripletta, saturazione di valenza
- La sostituzione avviene **solo** se il punteggio alternativo supera quello originale di almeno 1.5 punti (soglia conservativa)
- Questa fase **non aggiunge né rimuove nodi**: modifica solo l'etichetta di tipo di nodi esistenti

---

### Step 4.3 — Consenso Multi-Vista sui Tipi di Connessione

**Idea**: La prospettiva cambia le distanze pixel tra componenti, ma non cambia **quali tipi di componenti sono adiacenti**. Questo è il segnale più robusto disponibile.

**Processo** (`aggregate_type_consensus`):
- In ogni vista, si costruisce un grafo K-NN (K=3) sulle detections topologiche
- Gli archi di questo grafo rappresentano coppie di tipi adiacenti in quella vista
- Aggregando su tutte le viste, si ottiene per ogni coppia di tipi `(A, B)` la frazione di viste in cui appaiono adiacenti: `type_consensus[(A, B)] = n_viste_con_adiacenza / n_viste_totali`
- Il risultato è un dizionario `{(codice_A, codice_B): [0, 1]}` che rappresenta la **fiducia multi-vista** in quella coppia di tipi

Questo segnale è indipendente dalla geometria e cattura informazioni topologiche pure.

---

### Step 4.4 — Generazione dei Candidati Locali

**Idea**: Non si può costruire un grafo su tutte le possibili coppie di nodi (combinatorialmente esplosivo e semanticamente privo di senso). Si restringe il dominio di ricerca a coppie **geometricamente plausibili** nella vista migliore.

**Processo** (`_build_local_candidate_pairs`):

**A. Filtro di mutua vicinanza** (mutual-rank filtering):
Una coppia `(i, j)` è candidata solo se:
- `i` è tra i `LOCAL_TOP_K=4` vicini più prossimi di `j`
- `j` è tra i `LOCAL_RANK_K=5` vicini più prossimi di `i`

Questo garantisce che entrambi i nodi si "riconoscano" come vicini, evitando connessioni asimmetriche spurie.

**B. Gate di distanza**:
- Distanza massima assoluta: `cutoff × HARD_MAX_DIST_FACTOR (2.0)`
- Il `cutoff` è la distanza mediana tra vicini immediati della scena, moltiplicata per `DIST_MULTIPLIER=2.5`
- Per nodi hub (valenza ≥ 3) con forte supporto multi-vista: range esteso fino a `cutoff × 2.8`

**C. Trattamento speciale degli hub**:
I componenti ad alta valenza (Tee, Croce) hanno bisogno di più candidati. Il sistema applica:
- Tolleranza aggiuntiva di `+3` vicini e `+2` rank per coppie che coinvolgono hub
- "Hub rescue": se una coppia hub-preferita ha `mv_score ≥ 0.5`, il gate di distanza viene espanso

**D. Scoring dei candidati**:

```
score = mutual_rank × 3.5
      + mv_score × 3.0
      + cooc_score × 2.0
      + hub_bonus
      - hub_penalty
      - (dist_norm²) × 4.0
```

dove:
- `mutual_rank`: quanto i due nodi si trovano in cima alle liste di vicini reciproci
- `mv_score`: consenso multi-vista sul tipo di connessione `(A, B)`
- `cooc_score`: frequenza storica di co-occorrenza della coppia nel dataset (normalizzata)
- `hub_bonus/penalty`: bonus semantico se la coppia coinvolge vicini preferiti/sfavoriti degli hub
- Penalità quadratica sulla distanza normalizzata (connessioni lunghe sono scoraggiate, non vietate)

**Nota fondamentale**: Il multi-vista domina sulla geometria pura. Questo è intenzionale: una coppia che appare adiacente in tutte le viste ma è lontana nella vista migliore (per prospettiva) deve comunque essere considerata.

**Fallback**: Ogni nodo isolato ottiene almeno un candidato (il vicino più prossimo semanticamente consentito).

Il risultato tipico sono 20-40 coppie candidate per una scena di 10-15 componenti.

---

### Step 4.5 — Ottimizzazione ILP (Integer Linear Programming)

**Idea**: Trovare il miglior sottoinsieme di archi candidati che formi un **albero ricoprente** (spanning tree) rispettando i vincoli di valenza fisici di ogni nodo.

**Formulazione del problema** (`build_graph_ilp_soft`):

Variabili: `x[i,j] ∈ {0, 1}` — arco incluso o escluso

**Funzione obiettivo** (massimizzazione):
```
maximize Σ_{(i,j) in candidati} score[i,j] × x[i,j]
```

**Vincoli duri**:

1. **Vincolo di valenza superiore**: ogni nodo non può superare la propria valenza fisica
   ```
   Σ_j x[i,j] ≤ valence[i]   ∀ i
   ```

2. **Saturazione degli hub**: se un nodo hub ha abbastanza candidati, deve usare tutta la sua valenza
   ```
   Σ_j x[i,j] = valence[i]   per hub con candidati sufficienti
   ```

3. **Albero ricoprente**: il numero totale di archi deve essere esattamente `n-1`
   ```
   Σ_{i,j} x[i,j] = n - 1
   ```

4. **Connettività** (iterativa): dopo ogni soluzione, se il grafo risultante è disconnesso, si aggiungono vincoli di taglio:
   ```
   Σ_{e attraversa il taglio} x[e] ≥ 1
   ```
   e il problema viene risolto nuovamente. Si ripete fino alla connettività.

**Bonus a triplette** (linearizzati):
Prima dell'ottimizzazione, si pre-calcolano bonus aggiuntivi agli score degli archi basati su pattern a tripletta `(A - centro - B)` osservati nel dataset. Se un nodo centrale è di tipo `C` e i pattern `(A, C, B)` sono frequenti, gli archi `(centro, A)` e `(centro, B)` ricevono un bonus nel loro score ILP. Questo incoraggia strutture localmente coerenti con l'esperienza storica.

**Soluzione**: Il solver CBC (via libreria PuLP) trova la soluzione ottimale esatta. Il risultato è un insieme di archi `{(i,j)}` che forma un albero ricoprente massimo rispetto alla funzione obiettivo.

**Fallback**: Se l'ILP risulta infeasible (raro), si ricade su un algoritmo greedy (`build_graph_consensus_first`).

---

### Step 4.6 — Post-Processing degli Hub

**Idea**: Dopo l'ILP, si esegue un affinamento locale attorno ai nodi hub. L'ILP lavora globalmente ma può aver scelto per un hub un vicino geometricamente conveniente ma semanticamente subottimale, quando un candidato migliore era disponibile.

**Processo** (`_postprocess_tree_edges`):
- Per ogni nodo hub (valenza ≥ 3), si esamina se uno dei suoi archi nel grafo punta a un vicino "sfavorito" (es. Giunto, Riduzione, Bout bic.)
- Se esiste un candidato alternativo "preferito" (es. Saracinesca, Tee) non ancora connesso che soddisfa i vincoli di grado, e il guadagno di score è > 0.5, l'arco viene sostituito
- Ogni sostituzione è loggata come `[PostHub]`

Questa fase è minimale: non stravolge il grafo, ma corregge casi specifici in cui la prospettiva ha sfavorito un vicino semanticamente naturale.

---

### Step 4.7 — Conferma Multi-Vista degli Archi

**Idea**: Dopo la costruzione del grafo, si calcola quante viste "supportano" ogni arco finale. Questo non altera il grafo ma fornisce una metrica di fiducia per la visualizzazione.

**Processo** (`confirm_edges_multivista`):
- Per ogni arco `(u, v)` nel grafo finale, si controlla in ogni altra vista se il tipo di `v` appare tra i K vicini più prossimi del tipo di `u`
- `confirm_ratio = n_viste_che_confermano / n_altre_viste`

Nota: il confronto avviene sui **tipi** di componente, non sulle istanze. Non è tracking multi-vista completo, ma una stima della coerenza topologica.

---

### Step 4.8 — Rimozione dei Cicli (Safety Net)

**Idea**: Se l'ILP ha avuto successo, il grafo è già un albero per costruzione. Questa fase è un guardrail di sicurezza per il percorso di fallback.

**Processo** (`break_cycles`):
- Si rilevano cicli con DFS
- Si rimuove l'arco più debole nel ciclo usando uno scoring che considera: multi-vista, conferma, triplette, co-occorrenze, penalità varie
- Si verifica che nessun nodo resti isolato dopo la rimozione

---

## Base di Conoscenza (Priors Statistici)

La pipeline non lavora nel vuoto: sfrutta conoscenza appresi dal dataset in tre forme.

### Co-occorrenze (`COOC_COUNTS`)

Frequenza assoluta con cui ogni coppia di tipi di componenti è stata osservata connessa nel dataset di training. Esempio: (15, 95) = 21 osservazioni (Saracinesca ↔ Bout bic.). Normalizzato sul massimo per produrre `cooc_score ∈ [0, 1]`.

Usato come prior bayesiano: coppie frequenti sono favorite nel candidato scoring.

### Pattern a Tripletta (`TRIPLET_COUNTS`)

Strutture locali della forma `(vicino_A — nodo_centrale — vicino_B)` osservate nel dataset. Esempio: (15, 116, 15) = 12 occorrenze (Saracinesca - Tee FL - Saracinesca).

Usato in due punti:
1. Come bonus agli score nell'ILP (step 4.5)
2. Come segnale nella consolidazione dei codici (step 4.2)

Cattura la coerenza semantica locale: se un nodo è di tipo X, i suoi vicini tendono ad essere di tipi specifici.

### Saturazione di Valenza (`DEGREE_SATURATION`)

Frazione delle istanze nel dataset in cui un componente di un dato tipo è stato osservato con tutte le sue porte connesse visibili. Esempio: Giunto (code 11) = 0.00 → non è mai stato osservato con entrambi i lati visibili → nella pratica si comporta come un nodo terminale.

Usato in `effective_valence()`: un componente con bassa saturazione viene trattato con valenza effettiva ridotta (tipicamente 1) a meno che non ci sia forte supporto multi-vista per una seconda connessione. Questo evita di forzare connessioni che in realtà escono dal campo visivo.

### Coppie Vietate (`FORBIDDEN_PAIRS`)

112 coppie di tipi mai osservate connesse nel dataset. Usate come vincoli duri in `edge_allowed()`: nessuna coppia vietata può essere aggiunta come arco candidato, indipendentemente dalla geometria.

---

## Sistema di Componenti e Valenze

Il sistema conosce 31 tipi di componenti idraulici organizzati per valenza:

| Valenza | Tipo | Codici Esempio |
|---------|------|----------------|
| 1 | Terminali (es. Idrante) | 37 |
| 2 | In-line (valvole, giunti, curve, tubi) | 11, 15, 53, 54, 75, 95, 111, 175, 237, ... |
| 3 | Raccordi a T | 83, 85, 116, 159, 201, 232, 255 |
| 4 | Raccordi a croce | 61, 192 |
| 0 | Marker di diametro (non topologici) | 114, 125, 228, 231, 244, 247, 250, 252 |

I marker di diametro (valenza 0) vengono ignorati nella costruzione del grafo topologico.

---

## Output e Visualizzazione

Il grafo finale è rappresentato come struttura NetworkX dove:
- I **nodi** hanno attributi: codice tipo, nome, valenza fisica, valenza effettiva, posizione 2D
- Gli **archi** hanno attributi: distanza, score multi-vista, co-occorrenza, score tripletta, rank mutuo, conferma, flag di provenienza (ILP / post-processing / fallback)

### Semantica della Visualizzazione

Gli archi nel grafo overlay sono colorati secondo la loro fiducia:

| Colore | Stile | Significato |
|--------|-------|-------------|
| Ciano solido | `—` | Arco forzato + forte supporto semantico (mv ≥ 0.5 e cooc ≥ 0.5) |
| Ciano tratteggiato | `-.` | Arco forzato per connettività (bassa fiducia) |
| Magenta puntato | `:` | Arco di recovery (deprecato) |
| Rosa tratteggiato | `--` | Arco aggiunto dal post-processing hub |
| Verde | `—` | `confirm_ratio ≥ 0.5`: ben confermato multi-vista |
| Giallo | `—` | `confirm_ratio < 0.5` ma `cooc > 0.3`: prior forte |
| Arancione | `--` | "Dubbio": né alta conferma né co-occorrenza forte |

**Importante**: un arco arancione non è necessariamente sbagliato — è semplicemente incerto secondo le metriche di visualizzazione. Un arco può essere topologicamente corretto e ottimale nell'ILP pur avendo bassa conferma visiva.

---

## Principi Architetturali

La pipeline è costruita attorno a quattro principi fondamentali:

1. **Multi-vista > geometria singola**: Il consenso multi-vista è il segnale primario. La geometria restringe il campo di ricerca ma non decide.

2. **Geometria come filtro locale**: La distanza nella vista migliore determina quali coppie entrano come candidati. Non determina l'arco vincitore.

3. **Semantica come riranking**: I priors statistici (co-occorrenze, triplette) affinano lo scoring all'interno del vicinato locale già filtrato geometricamente.

4. **Ottimizzazione globale per la struttura**: L'ILP garantisce simultaneamente valenza fisica, proprietà di albero e massima plausibilità complessiva. Nessun algoritmo greedy locale può garantire questo.

---

## Limitazioni Note

- **Distorsione prospettica nella vista migliore**: se un ramo corretto appare lontano per prospettiva, l'ILP potrebbe preferire un vicino geometricamente più vicino ma errato. Mitigato da hub rescue e post-processing.
- **Corrispondenza multi-vista per tipo, non per istanza**: la conferma multi-vista lavora sui tipi di componente, non traccia le singole istanze attraverso le viste. Questo è un'approssimazione.
- **Qualità del fallback greedy**: se l'ILP è infeasible, la soluzione greedy è meno precisa.

---

## Schema Riassuntivo del Flusso

```
Immagini multi-vista
        |
        v
[Step 1] YOLO + CLAHE → Shurikode decoder → Cache JSON detections
        |
        v
[Step 4.1] Score viste → Selezione vista migliore (geometria di riferimento)
        |
        v
[Step 4.2] Consolidamento codici vista migliore (consenso inter-vista)
        |
        v
[Step 4.3] K-NN per vista → Aggregazione → type_consensus[(A,B)] ∈ [0,1]
        |
        v
[Step 4.4] Mutual-rank + gate distanza + hub-rescue → Candidati + score
        |
        v
[Step 4.5] ILP (massimizzazione score | vincoli valenza + albero + connettività)
        |
        v
[Step 4.6] Post-processing hub (rewiring locale minimale)
        |
        v
[Step 4.7] Conferma multi-vista archi (solo annotazione)
        |
        v
[Step 4.8] Safety net rimozione cicli
        |
        v
Grafo aciclico connesso con attributi e visualizzazione overlay
```

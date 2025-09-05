
<!-- README.md is generated from README.Rmd. Please edit that file -->

# APRENDIZADO DE MÁQUINA ESTATÍSTICO PARA ESTIMATIVA DA EMISSÃO DE CO<sub>2</sub> DO SOLO EM ÁREAS AGRÍCOLAS

**Beneficiário**: Luis Felipe Trevelim

**Responsável**: Alan Rodrigo Panosso

**Resumo**: A concentração de gases de efeito estufa (GEE) na atmosfera,
como o dióxido de carbono (CO<sub>2</sub>), aumentou consideravelmente
devido a fontes antropogênicas. No Brasil, atividades agrícolas e
florestais contribuem substancialmente para as emissões de
CO<sub>2</sub>, principalmente devido ao desmatamento e à conversão de
florestas nativas. Estudos anteriores demonstraram que FCO2 pode ser
modelada com alta precisão usando uma grande quantidade de variáveis
ambientais. No entanto, a conversão a longo prazo de florestas nativas
para agroecossistemas ainda é pouco compreendida, especialmente no
contexto brasileiro. Assim, a hipótese central é que as mudanças no uso
da terra para fins agrícolas alteram os atributos químicos e físicos do
solo, induzindo mudanças na emissão de CO2. Este projeto visa investigar
a emissão de CO<sub>2</sub> do solo (FCO2) em áreas agrícolas do bioma
Cerrado, utilizando técnicas de aprendizado de máquina estatístico para
modelar FCO2 com base em demais variáveis associadas.

**Palavras-chaves**: respiração do solo, inteligência artificial,
mudanças climáticas, aprendizado de máquina.

### [1-Faxina](https://arpanosso.github.io/projeto-trevelimlf/Docs/faxina.html)

### [2-Importação e Tratamento](https://arpanosso.github.io/projeto-trevelimlf/Docs/importacao_tratamento.html)

### 3 - Aprendizado de Máquina

#### Carregando os pacotes

``` r
library(tidyverse)
library(patchwork)
library(ggspatial)
library(readxl)
library(skimr)
library(tidymodels)
library(ISLR)
library(modeldata)
library(vip)
library(ggpubr)
theme_set(theme_bw())
```

#### Entrando com o banco de dados

``` r
data_set <- read_rds("data/data-set.rds")
glimpse(data_set)
```

##### Extraindo o grupo de variáveis

``` r
time_var <- data_set |> select(data:month) |> names()
catego_var <- data_set |> select(cultura, manejo, tratamento) |> names()
din_var <- data_set |> select(fco2:us,pla) |> names()
chemical_var <- data_set |> select(ph:v,hlifs) |> names()
physical_var <- data_set |> select(ds:vtp) |> names()
textural_var <- data_set |> select(at:arg) |> names()
textural_var <- data_set |> select(at:arg) |> names()
orbital_var <- data_set |> select(xco2_trend:sif) |> names()
meteorological_var <- data_set |> select(tmed:inso) |> names()
```

### Dividindo a base entre treino e teste

``` r
fco2_initial_split <- initial_split(data_set, prop = 0.80)
fco2_train <- training(fco2_initial_split)
# fco2_test <- testing(fco2_initial_split)
# visdat::vis_miss(fco2_test)
fco2_train  |>  
  ggplot(aes(x=fco2, y=..density..))+
  geom_histogram(bins = 30, color="black",  fill="lightgray")+
  geom_density(alpha=.05,fill="red")+
  theme_bw() +
  labs(x="fco2 - treino", y = "Densidade")
```

``` r
fco2_testing <- testing(fco2_initial_split)
fco2_testing  |>  
  ggplot(aes(x=fco2, y=..density..))+
  geom_histogram(bins = 30, color="black",  fill="lightgray")+
  geom_density(alpha=.05,fill="blue")+
  theme_bw() +
  labs(x="fco2 - teste", y = "Densidade")
```

### Definindo a Reamostragem

``` r
fco2_resamples <- vfold_cv(fco2_train, v = 5)
```

``` r

fco2_recipe <- recipe(fco2 ~ ., 
                      data = fco2_train |>  
            select(fco2:inso, -xco2_trend) # 
) |>   
  # step_naomit(all_outcomes()) |>  # remove linhas sem fco2  
  step_naomit(c(ts, us)) |>  # retira NAs somente de ts e us
  # step_novel(all_nominal_predictors()) |>  # evitar problemas quando aparece categoria nova
  step_zv(all_predictors()) |> # evita problemas com variância zero 
  # step_poly(c(us,ts), degree = 2)  |> #polinômios de us e ts de grau 2  
  step_impute_median(all_numeric_predictors()) |>  # inputação da mediana - antes de normalize
  step_normalize(all_numeric_predictors())  #|>   padronização normal (x-mu)/sigmma
  # step_dummy(all_nominal_predictors()) # converte fatores em variáveis binárias
bake(prep(fco2_recipe), new_data = NULL)
```

## SUPPORT VECTOR MACHINE - RDF

#### ϵ-insensitive loss regression (Flavor).

<https://bradleyboehmke.github.io/HOML/svm.html>
<https://stackoverflow.com/questions/77735850/variable-importance-plot-for-support-vector-machine-with-tidymodel-framework-is>

#### Definição do Modelo de Função de Base Radial

#### Definir os parâmetros da tunagem

``` r
fco2_svm_model <- svm_rbf(
  cost = tune(), 
  rbf_sigma = tune(), 
  margin = tune()) |>  # margin sempre para regressão -->
  set_mode("regression") |> 
  set_engine("kernlab") #%>% -->
 #translate() -->
```

#### Workflow e tunagem

``` r
fco2_svm_wf <- workflow()   |> 
  add_model(fco2_svm_model) |> 
  add_recipe(fco2_recipe)

grid_svm <- expand.grid( 
  cost = c(0.01), #0.0625, 0.1, 1, 10, 20,
  rbf_sigma = c(0.001),  #0.095,
  margin = c(-3,-2) # 0.025,
)
glimpse(grid_svm)

fco2_svm_tune_grid <- tune_grid(
  fco2_svm_wf,
  resamples = fco2_resamples,
  grid = grid_svm,
  metrics = metric_set(rmse)
)
autoplot(fco2_svm_tune_grid)
```

### Coletando métricas

``` r
collect_metrics(fco2_svm_tune_grid)
fco2_svm_tune_grid |> 
  show_best(metric = "rmse", n = 6)
```

### Desempenho do modelo final

``` r
fco2_svm_best_params <- select_best(fco2_svm_tune_grid, metric = "rmse")
fco2_svm_wf <- fco2_svm_wf |> 
  finalize_workflow(fco2_svm_best_params)
fco2_svm_last_fit <- last_fit(fco2_svm_wf, fco2_initial_split)

## Criando os preditos
fco2_test_preds <- bind_rows(
  collect_predictions(fco2_svm_last_fit)  |> 
    mutate(modelo = "svm"))

fco2_test <- testing(fco2_initial_split)

fco2_test_preds |> 
  ggplot(aes(x=.pred, y=fco2)) +
  geom_point()+
  theme_bw() +
  geom_smooth(method = "lm") +
  stat_regline_equation(ggplot2::aes(
  label =  paste(..eq.label.., ..rr.label.., sep = "*plain(\",\")~~"))) +
  geom_abline (slope=1, linetype = "dashed", color="Red")
```

## Salvando o modelo final

``` r
fco2_modelo_final <- fco2_svm_wf |> 
  fit(data_set)
saveRDS(fco2_modelo_final, "models/fco2_modelo_svm_.rds")
```

``` r
# Extract the actual training data from your workflow
training_data <- fco2_svm_last_fit$.workflow[[1]]$pre$mold$predictors
training_target <- fco2_svm_last_fit$.workflow[[1]]$pre$mold$outcomes$fco2

# First, create the vip plot and store it
vip_plot <- fco2_modelo_final |> 
  extract_fit_parsnip() |>  
  vip( 
    method = "permute", 
    target = "fco2", 
    metric = "rmse", 
    nsim = 5, 
    pred_wrapper = function(object, newdata) {
      workflow_pred <- fco2_svm_last_fit$.workflow[[1]]
      predict(workflow_pred, newdata) %>% pull(.pred)
    },
    train = fco2_train,
    aesthetics = list(color = "black", fill = "orange")) + 
  theme(axis.text.y=element_text(size=rel(1.5)), 
        axis.text.x=element_text(size=rel(1.5)), 
        axis.title.x=element_text(size=rel(1.5))
  )
```

``` r
importance_top_10 <- vip_plot$data

importance_top_10 |> 
  mutate(feature_type = case_when(
    Variable %in% physical_var   ~ "físicos",
    Variable %in% chemical_var  ~ "químicos",
    Variable %in% din_var ~ "dinâmicos",
    Variable %in% meteorological_var ~ "climáticos",
    Variable %in% orbital_var  ~ "orbitais",
    Variable %in% textural_var  ~ "textura",
    Variable %in% time_var  ~ "tempo",
    TRUE                        ~ "manejo"
  ),
  Variable = Variable |> fct_reorder(Importance)) |> 
  ggplot(aes(x=Importance, y=Variable, fill = feature_type)) +
  geom_col(color="black") +
  theme_bw()+
  labs(x = "Importância",y="",
       fill="Grupo") +
  theme(legend.position = "top") +
  scale_fill_viridis_d()

fco2_nn_last_fit_model$censor_probs |> str()
```

### Principais Métricas

``` r
da <- fco2_test_preds |> 
  filter(fco2 > 0, .pred>0 )

my_r <- cor(da$fco2,da$.pred)
my_r2 <- my_r*my_r
my_mse <- Metrics::mse(da$fco2,da$.pred)
my_rmse <- Metrics::rmse(da$fco2,
                         da$.pred)
my_mae <- Metrics::mae(da$fco2,da$.pred)
my_mape <- Metrics::mape(da$fco2,da$.pred)*100

vector_of_metrics <- c(r=my_r, R2=my_r2, MSE=my_mse, RMSE=my_rmse, MAE=my_mae, MAPE=my_mape)
print(data.frame(vector_of_metrics))
#>      vector_of_metrics
#> r            0.6787708
#> R2           0.4607298
#> MSE          0.1984555
#> RMSE         0.4454834
#> MAE          0.3259117
#> MAPE        25.2042723
```

<!--
## REDE NEURAL ARTIFICIAL
#### Definição do Modelo de RNA - MultiLayer Perceptron
&#10;``` r
fco2_nn_model <- mlp() |>  # margin sempre para regressão
  set_mode("regression") |> 
  set_engine("nnet")
```
&#10;#### Definir os parâmetros da tunagem
&#10;
``` r
fco2_nn_model <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
  ) |>  # margin sempre para regressão
  set_mode("regression") |> 
  set_engine("nnet")
```
&#10;#### Workflow e tunagem
&#10;
``` r
fco2_nn_wf <- workflow()   |> 
  add_model(fco2_nn_model) |> 
  add_recipe(fco2_recipe)
# Criando a matriz (grid) com os valores de hiperparâmetros a serem testados
# grid_nn <- expand.grid(
#   hidden_units = c(1,5,6,7,8,15), #c(1,5,6,7,8,15)
#   penalty = c(1,20),  #c(1,5,10,20)
#   epochs = c(50,1000) # c(50,100,500,1000)
# )
&#10;grid_nn <- grid_regular(
  hidden_units(range = c(10, 60)), ## tentar até 250
  penalty(range = c(-5, 10), trans = scales::log10_trans()), ## no máximo 30
  epochs(range = c(75, 80)),
  levels = c(3, 3, 3)
)
glimpse(grid_nn)
&#10;fco2_nn_tune_grid <- tune_grid(
  fco2_nn_wf,
  resamples = fco2_resamples,
  grid = grid_nn,
  metrics = metric_set(rmse)
)
autoplot(fco2_nn_tune_grid)
```
&#10;### Coletando métricas
&#10;``` r
collect_metrics(fco2_nn_tune_grid)
fco2_nn_tune_grid |> 
  show_best(metric = "rmse", n = 6)
```
&#10;### Desempenho do modelo final
&#10;``` r
fco2_nn_best_params <- select_best(fco2_nn_tune_grid, metric = "rmse")
fco2_nn_wf <- fco2_nn_wf |> 
  finalize_workflow(fco2_nn_best_params)
fco2_nn_last_fit <- last_fit(fco2_nn_wf, fco2_initial_split)
&#10;## Criando os preditos
fco2_test_preds <- bind_rows(
  collect_predictions(fco2_nn_last_fit)  |> 
    mutate(modelo = "nn"))
&#10;fco2_test <- testing(fco2_initial_split)
&#10;fco2_test_preds |> 
  ggplot(aes(x=.pred, y=fco2)) +
  geom_point()+
  theme_bw() +
  geom_smooth(method = "lm") +
  stat_regline_equation(ggplot2::aes(
  label =  paste(..eq.label.., ..rr.label.., sep = "*plain(\",\")~~"))) +
  geom_abline (slope=1, linetype = "dashed", color="Red")
```
&#10;## Salvando o modelo final
&#10;``` r
fco2_modelo_final <- fco2_nn_wf |> 
  fit(data_set)
saveRDS(fco2_modelo_final, "models/fco2_modelo_nn_.rds")
```
&#10;
&#10;``` r
fco2_nn_last_fit_model <- fco2_nn_last_fit$.workflow[[1]]$fit$fit
vip(fco2_nn_last_fit_model,
    aesthetics = list(color = "black", fill = "orange")) +
    theme(axis.text.y=element_text(size=rel(1.5)),
          axis.text.x=element_text(size=rel(1.5)),
          axis.title.x=element_text(size=rel(1.5))
          ) +
  theme_bw()
```
&#10;
``` r
importance_top_10 <- vi(fco2_nn_last_fit_model) |> 
  arrange(desc(Importance)) |> 
  slice(1:10)
&#10;importance_top_10 |> 
  mutate(feature_type = case_when(
    Variable %in% physical_var   ~ "físicos",
    Variable %in% chemical_var  ~ "químicos",
    Variable %in% din_var ~ "dinâmicos",
    Variable %in% meteorological_var ~ "climáticos",
    Variable %in% orbital_var  ~ "orbitais",
    Variable %in% textural_var  ~ "textura",
    Variable %in% time_var  ~ "tempo",
    TRUE                        ~ "manejo"
  ),
  Variable = Variable |> fct_reorder(Importance)) |> 
  ggplot(aes(x=Importance, y=Variable, fill = feature_type)) +
  geom_col(color="black") +
  theme_bw()+
  labs(x = "Importância",y="",
       fill="Grupo") +
  theme(legend.position = "top") +
  scale_fill_viridis_d()
&#10;fco2_nn_last_fit_model$censor_probs |> str()
&#10;```
&#10;
### Principais Métricas
&#10;
``` r
da <- fco2_test_preds |> 
  filter(fco2 > 0, .pred>0 )
&#10;my_r <- cor(da$fco2,da$.pred)
my_r2 <- my_r*my_r
my_mse <- Metrics::mse(da$fco2,da$.pred)
my_rmse <- Metrics::rmse(da$fco2,
                         da$.pred)
my_mae <- Metrics::mae(da$fco2,da$.pred)
my_mape <- Metrics::mape(da$fco2,da$.pred)*100
&#10;vector_of_metrics <- c(r=my_r, R2=my_r2, MSE=my_mse, RMSE=my_rmse, MAE=my_mae, MAPE=my_mape)
print(data.frame(vector_of_metrics))
#>      vector_of_metrics
#> r            0.6787708
#> R2           0.4607298
#> MSE          0.1984555
#> RMSE         0.4454834
#> MAE          0.3259117
#> MAPE        25.2042723
```
-->

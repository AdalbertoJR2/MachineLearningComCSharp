using MachineLearning.ML;

ExemploRegressao();

void ExemploRegressao()
{
    var trainer = new CasaModelTrainer();
    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "casas_treinamento.csv"));
}
using MachineLearning.ML;

ExemploRegressao();

void ExemploRegressao()
{
    var trainer = new CasaModelTrainer();
    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "casas_treinamento.csv"));
    trainer.TreinarModelo();
    
    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinamento_regrassao.zip");
    trainer.SalvarModelo(pathModelo);
}
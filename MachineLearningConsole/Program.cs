using MachineLearning.ML;
using MachineLearning.Models;

ExemploRegressao();

void ExemploRegressao()
{
    //Treina o modelo.
    var trainer = new CasaModelTrainer();
    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "casas_treinamento_grande.csv"));
    trainer.TreinarModelo();
    trainer.AvaliarModelo();
    
    //Salva o modelo treinado e sobrescreve o que já está salvo.
    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinamento_regrassao.zip");
    trainer.SalvarModelo(pathModelo);

    //Carregando e testando a predição de dados.
    var predictor = new CasaModelPredictor();
    predictor.CarregarModelo(pathModelo);

    var casaNova = new CasaInputData()
    {
        Tamanho = 85f,
        Quartos = 3
    };

    var resultado = predictor.Prever(casaNova);
    Console.WriteLine(  "O valor da casa nova é: " + resultado.PrecoPrevisto );
}
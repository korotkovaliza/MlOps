# ОПРЕДЕЛЕНИЕ ФЕЙКОВЫХ НОВОСТЕЙ
## Формулировка задачи
Моя задача - использовать глубокое обучение и преобразование данных через Embedding для автоматического обнаружения фейковых новостей в социальных сетях. Это необходимо для повышения надежности в идентификации фейковых новостей и использования паттернов распространения в социальных сетях в качестве ключевых признаков.

## Данные
Набор данных [LIAR](https://github.com/tfs4/liar_dataset) состоит из 12 836 коротких высказываний, взятых из POLITIFACT со следующими столбцами: идентификатор высказывания ([ID].json), метка, высказывание, тема, говорящий, должность говорящего, информация о штате, партийная принадлежность, общее количество проверенных фактов в истории(включая текущее высказывание) и контекст(место/местоположение выступления или высказывания). 
Пример датасета с фейковыми gossipcop новостями:
![Image alt](https://github.com/korotkovaliza/MlOps/blob/main/dataset1.png)
Для высказываний в наборе данных имеется шесть меток: pants-fire, false, mostly-false, half-true, mostly-true, and true. Эти шесть наборов меток относительно сбалансированы по размеру. Утверждения были собраны из различных средств вещания, таких как телеинтервью, речи, твиты, дебаты, и охватывают широкий спектр тем, таких как экономика, здравоохранение, налоги и выборы.

## Подход к моделированию
В модели используется гибрид CNN для утверждений и LSTM для темы, работы говорящего, контекста и обоснования.
Вместо того чтобы напрямую извлекать признаки из высказывания, мы используем механизм внимания, чтобы использовать заданную побочную информацию (субъект, спикер, должность, штат, партия, контекст и обоснование) для проверки правдивости высказывания. Механизм внимания делает процесс извлечения признаков из высказывания контекстуализированным на основе побочной информации. 
Графическое представление архитектуры:
![Image alt](https://github.com/korotkovaliza/MlOps/blob/main/fake-net.png)


## Способ предсказания
После обучения модели необходимо будет реализовать продакшен пайплайн, который включает в себя предобработку данных, запуск модели на новых данных, и вывод предсказаний. Финальное применение модели может быть встроено в информационные потоки социальных медиа для автоматического обнаружения фейковых новостей. Архитектурную схему продакшен пайплайна можно представить в виде последовательных блоков обработки данных, подачи данных в модель и интерпретации предсказаний.

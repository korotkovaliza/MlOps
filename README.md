# ОПРЕДЕЛЕНИЕ ФЕЙКОВЫХ НОВОСТЕЙ
## Формулировка задачи
Моя задача - использовать глубокое обучение и геометрическую обработку данных для автоматического обнаружения фейковых новостей в социальных сетях. Это необходимо для повышения надежности в идентификации фейковых новостей и использования паттернов распространения в социальных сетях в качестве ключевых признаков.

## Данные
Для решения задачи буду использовать набор данных, собранный из новостных источников, верифицированных профессиональными организациями по факт-чекингу, такими как Snopes, PolitiFact и Buzzfeed. Эти данные содержат новости, как правдивые, так и ложные, которые распространялись в Twitter с 2013 по 2018 годы. Особенностью данных является их разнообразие и достоверность меток истинности. Ознакомиться с датасетами можно [тут](https://github.com/KaiDMML/FakeNewsNet/tree/master/dataset).
Пример датасета с фейковыми gossipcop новостями:
![Image alt](https://github.com/korotkovaliza/MlOps/blob/main/gossipcop.jpg)

## Подход к моделированию
Планирую использовать глубокое обучение на графах, включая четырехслойную графовую сверточную нейронную сеть с использованием графовой аттенции. Модель будет обучаться на различных признаках, описывающих пользователей, новости, их распространение и социальный граф. И будет использоваться библиотека глубокого обучения PyTorch. Схема модели представлена на блок-схеме ниже:

![Image alt](https://github.com/korotkovaliza/MlOps/blob/main/model.png)

## Способ предсказания
После обучения модели необходимо будет реализовать продакшен пайплайн, который включает в себя предобработку данных, запуск модели на новых данных, и вывод предсказаний. Финальное применение модели может быть встроено в информационные потоки социальных медиа для автоматического обнаружения фейковых новостей. Архитектурную схему продакшен пайплайна можно представить в виде последовательных блоков обработки данных, подачи данных в модель и интерпретации предсказаний.

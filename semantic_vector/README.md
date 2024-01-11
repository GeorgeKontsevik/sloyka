Структура:
1. models - директория для хранения моделей
2. output_data - директория для результатов оценки симантической близости
3. training data - директория для хранения данных для тренировки модели (пока что граф лежит там же)
4. src - исполняемые модули:
    На данный момент описаны исключительно в виде скрипта (позде перепишу в классы, но нужна консультация)
    1. get_static_model.py - Модуль скачивает модель с с сайта https://rusvectores.org/ru/models/
    2. create_custom_model.py - Создаёт пользовательскую модель. Пока что только на основе наштх данных. 
    3. use_static_model.py - Загружает и проверяет модель KeyedVectors (нельзя дообучить).
    4. evaluate_graph.py - Оценивает близость узлов в графе.
    5. 

Логика заключается в разделении функция статичной модели и обучаемой. Не знаю, насколько это правильно.

Проблемы, с которыми столкнулся.
Во-первых, триплеты из графа знаний не всегда отдельные слова. Может прийти несколько слов, их модель оценить не может, только отдельные слова в силу особенностей обучения (текст для обучения разбивается на отдельные слова).
Во-вторых, выдаёт ошибку с английскими словами (например, mercedes). Возможно, проблема решаема, пока что просто решил их не включать в анализ.
В-третьих, в статичной моедели не представлены слова во всех подежах, что не позволяет использовать её полноценно.

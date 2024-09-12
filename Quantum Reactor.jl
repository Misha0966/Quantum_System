using LinearAlgebra  # Импортирует модуль для линейной алгебры
using Random  # Импортирует модуль для генерации случайных чисел
using Distributions  # Импортирует модуль для работы с распределениями
using Plots  # Импортирует модуль для построения графиков
using Printf  # Импортирует модуль для форматированного вывода
using Flux  # Импортирует модуль для машинного обучения
using Base.Threads  # Импортирует модуль для многопоточности
using Dates  # Импортирует модуль для работы с датами и временем

gr()  # Инициализирует графический интерфейс для построения графиков

# Физические константы
const HBAR = 1.0545718e-34  # Постоянная Планка деленная на 2π
const M_E = 9.10938356e-31  # Масса электрона
const E = 1.60217662e-19  # Заряд электрона
const EPSILON_0 = 8.854187817e-12  # Электрическая постоянная

# Структура для хранения энергии
mutable struct EnergyStorage
    capacity::Float64  # Вместимость накопителя
    current_energy::Float64  # Текущая энергия в накопителе

    function EnergyStorage(capacity::Float64)
        new(capacity, 0.0)  # Инициализирует накопитель с нулевой начальной энергией
    end
end

# Функции для управления энергией в накопителе
function charge!(storage::EnergyStorage, amount::Float64)
    storage.current_energy = min(storage.capacity, storage.current_energy + amount)  # Заряжает накопитель, не превышая его вместимость
end

function discharge!(storage::EnergyStorage, amount::Float64)
    if amount > storage.current_energy
        error("Недостаточно энергии для разрядки.")  # Выдает ошибку, если запрашиваемое количество энергии больше текущего
    end
    storage.current_energy -= amount  # Разряжается накопитель
end

function get_current_energy(storage::EnergyStorage)
    return storage.current_energy  # Возвращает текущую энергию накопителя
end

# Структура для распределения энергии
mutable struct EnergyDistributor
    efficiency::Float64  # Эффективность распределения энергии

    function EnergyDistributor(efficiency::Float64)
        new(efficiency)  # Инициализирует распределитель с заданной эффективностью
    end
end

# Функция распределения энергии
function distribute_energy!(distributor::EnergyDistributor, storage::EnergyStorage, amount::Float64)
    energy_to_distribute = amount * distributor.efficiency  # Рассчитывает количество энергии для распределения
    charge!(storage, energy_to_distribute)  # Заряжает накопитель распределенной энергией
end

# Функции для квантовой механики
function create_superposition_state(dim::Int)
    state = rand(ComplexF64, dim)  # Создает случайное состояние
    return state / norm(state)  # Нормализует состояние
end

function density_matrix(state::Vector{ComplexF64})
    return state * state'  # Возвращает плотностную матрицу состояния
end

function normalize_probabilities(probabilities::Vector{Float64})
    total = sum(probabilities)  # Суммирует вероятности
    return probabilities / total  # Нормализует вероятности
end

function measure_state(rho::Matrix{ComplexF64})
    probabilities = real(diag(rho))  # Извлекает вероятности из плотностной матрицы
    probabilities = normalize_probabilities(probabilities)  # Нормализует вероятности

    if any(probabilities .< 0)
        error("Некоторые значения вероятностей отрицательны.")  # Проверяет на отрицательные вероятности
    end

    if abs(sum(probabilities) - 1) > 1e-10
        error("Сумма вероятностей не равна 1.")  # Проверяет, что сумма вероятностей равна 1
    end

    dist = Categorical(probabilities)  # Создает категориальное распределение
    return rand(dist), probabilities  # Возвращает случайное состояние и вероятности
end

function pauli_x(dim::Int)
    op = Matrix{ComplexF64}(I, dim, dim)  # Создает матрицу Паули X
    for i in 1:dim
        op[i,i] = 0.0  # Обнуляет диагональные элементы
    end
    op[1,2] = 1.0  # Устанавливает элементы для X
    op[2,1] = 1.0  # Устанавливает элементы для X
    return op
end

function pauli_y(dim::Int)
    op = Matrix{ComplexF64}(I, dim, dim)  # Создает матрицу Паули Y
    for i in 1:dim
        op[i,i] = 0.0  # Обнуляет диагональные элементы
    end
    op[1,2] = im  # Устанавливает элементы для Y
    op[2,1] = -im  # Устанавливает элементы для Y
    return op
end

function pauli_z(dim::Int)
    op = Matrix{ComplexF64}(I, dim, dim)  # Создает матрицу Паули Z
    for i in 1:dim
        op[i,i] = 1.0 - 2.0 * i % 2  # Устанавливает элементы для Z
    end
    return op
end

function ising_hamiltonian(dim::Int, coupling::Float64, external_field::Float64)
    H = zeros(ComplexF64, dim, dim)  # Инициализирует Гамильтониан
    for i in 1:dim
        H += coupling * pauli_z(dim) * pauli_z(dim)  # Добавляет взаимодействие между спинами
        H += external_field * pauli_x(dim)  # Добавляет внешнее поле
    end
    return H
end

function external_field_hamiltonian(dim::Int, external_field::Float64)
    H = zeros(ComplexF64, dim, dim)  # Инициализирует Гамильтониан внешнего поля
    for i in 1:dim
        H[i, i] = external_field  # Устанавливает элементы для внешнего поля
    end
    return H
end

function create_entangled_state(dim::Int)
    state1 = create_superposition_state(dim)  # Создает первое запутанное состояние
    state2 = create_superposition_state(dim)  # Создает второе запутанное состояние
    entangled_state = kron(state1, state2)  # Создает запутанное состояние
    return entangled_state / norm(entangled_state)  # Нормализует запутанное состояние
end

function plot_state(probabilities::Vector{Float64}, title::String)
    bar(probabilities, title=title, xlabel="Индекс", ylabel="Вероятность", legend=false)  # Создает гистограмму вероятностей
end

function time_dependent_energy(start_energy::Float64, final_energy::Float64, current_step::Int, total_steps::Int)
    return start_energy * ((final_energy / start_energy) ^ (current_step / total_steps))  # Вычисляет энергию в зависимости от времени
end

function create_lindblad_operators(dim::Int)
    c_ops = Matrix{ComplexF64}[]  # Инициализирует массив операторов Линдблада
    for i in 1:(dim-1)
        for j in (i+1):dim
            op_matrix = zeros(ComplexF64, dim, dim)  # Создает матрицу оператора Линдблада
            op_matrix[i, j] = 1.0  # Устанавливает элементы
            push!(c_ops, op_matrix)  # Добавляет оператор в массив
        end
    end
    for i in 1:dim
        op_matrix = zeros(ComplexF64, dim, dim)  # Создает матрицу оператора Линдблада
        op_matrix[i, i] = 1.0  # Устанавливает элементы
        push!(c_ops, op_matrix)  # Добавляет оператор в массив
    end
    return c_ops  # Возвращает массив операторов Линдблада
end

function evolve_density_operator(rho::Matrix{ComplexF64}, H::Matrix{ComplexF64}, dt::Float64, c_ops::Vector{Matrix{ComplexF64}}, gamma::Float64)
    H_exp = exp(-im * H * dt)  # Вычисляет экспоненту Гамильтониана
    rho_evolved = H_exp * rho * H_exp'  # Эволюционирует плотностную матрицу

    Threads.@threads for c_op in c_ops
        c_op_term = c_op * rho_evolved * c_op' - 0.5 * (c_op' * c_op * rho_evolved + rho_evolved * c_op' * c_op)  # Рассчитывает вклад каждого оператора Линдблада
        rho_evolved += gamma * c_op_term  # Добавляет вклад оператора Линдблада
    end

    rho_evolved = (rho_evolved + rho_evolved') / 2  # Симметризует плотностную матрицу
    return rho_evolved  # Возвращает эволюционированную плотностную матрицу
end

# Функция для логирования данных
function log_data(step::Int, energy::Float64, probabilities::Vector{Float64}, rho::Matrix{ComplexF64}, filename::String)
    open(filename, "a") do file
        println(file, "Шаг $step:")  # Логирует текущий шаг
        println(file, "Энергия: $energy")  # Логирует энергию
        println(file, "Вероятности: $probabilities")  # Логирует вероятности
        println(file, "Плотностная матрица: \n$(string(rho))")  # Логирует плотностную матрицу
        
        # Дополнительные метрики
        mean_prob = mean(probabilities)  # Рассчитывает среднее значение вероятностей
        var_prob = var(probabilities)  # Рассчитывает дисперсию вероятностей
        std_dev_prob = sqrt(var_prob)  # Рассчитывает среднеквадратическое отклонение вероятностей
        ci_low, ci_high = confidence_interval(probabilities, 0.95)  # Рассчитывает доверительный интервал

        println(file, "Среднее значение вероятностей: $mean_prob")  # Логирует среднее значение
        println(file, "Дисперсия вероятностей: $var_prob")  # Логирует дисперсию
        println(file, "Среднеквадратическое отклонение вероятностей: $std_dev_prob")  # Логирует среднеквадратическое отклонение
        println(file, "95% Доверительный интервал: [$ci_low, $ci_high]")  # Логирует доверительный интервал
        println(file, "======================")  # Разделитель
    end
end

# Функция для расчета доверительного интервала
function confidence_interval(data::Vector{Float64}, confidence::Float64)
    mean_val = mean(data)  # Рассчитывает среднее значение данных
    stderr = std(data) / sqrt(length(data))  # Рассчитывает стандартную ошибку
    z_score = quantile(Normal(0, 1), (1 + confidence) / 2)  # Рассчитывает z-оценку для доверительного интервала
    margin_of_error = z_score * stderr  # Рассчитывает погрешность
    return (mean_val - margin_of_error, mean_val + margin_of_error)  # Возвращает доверительный интервал
end

# Функция для анализа данных
function analyze_data(step::Int, probabilities::Vector{Float64})
    if any(probabilities .< 0)
        error("Некоторые значения вероятностей отрицательны.")  # Проверяет на отрицательные вероятности
    end
    
    # Расчет метрик
    max_prob = maximum(probabilities)  # Рассчитывает максимальную вероятность
    mean_prob = mean(probabilities)  # Рассчитывает среднее значение вероятностей
    var_prob = var(probabilities)  # Рассчитывает дисперсию вероятностей
    std_dev_prob = sqrt(var_prob)  # Рассчитывает среднеквадратическое отклонение

    if max_prob > 0.5
        println("Внимание: высокая вероятность для одного состояния на шаге $step: $max_prob")  # Выдает предупреждение при высокой вероятности
    end

    return (max_prob > 0.5, mean_prob, var_prob, std_dev_prob)  # Возвращает метрики
end

# Функция для регулировки параметра gamma
function adjust_parameters!(gamma::Float64, high_prob_detected::Bool)
    previous_gamma = gamma  # Сохраняет предыдущее значение gamma
    if high_prob_detected
        gamma *= 1.1  # Увеличивает gamma при высокой вероятности
        println("Увеличиваем gamma: $gamma")  # Выводит сообщение об увеличении
    else
        gamma *= 0.9  # Уменьшает gamma при низкой вероятности
        println("Уменьшаем gamma: $gamma")  # Выводит сообщение о уменьшении
    end
    
    # Проверка, чтобы gamma не была отрицательной и не превышала разумные пределы
    gamma = clamp(gamma, 0.0, 1.0)  # Ограничивает gamma в пределах [0, 1]
    if gamma == 0.0 || gamma == 1.0
        println("Предупреждение: gamma достигла крайних значений ($gamma).")  # Выводит предупреждение о крайних значениях
    end
    
    # Возврат к предыдущему значению, если оно вышло за допустимый диапазон
    if gamma < 0.0 || gamma > 1.0
        gamma = previous_gamma  # Откатывает значение gamma
        println("Откат к предыдущему значению gamma: $gamma")  # Выводит сообщение об откате
    end
    
    return gamma  # Возвращает скорректированное значение gamma
end

# Функция для машинного обучения и регулировки gamma
function machine_learning_adjustment!(gamma::Float64, model::Flux.Chain, data::Vector{Float64})
    # Валидация данных для модели машинного обучения
    if length(data) != 16
        error("Некорректное количество данных для модели машинного обучения.")  # Проверка корректности данных
    end
    
    predictions = model(data)  # Получает предсказания модели
    predicted_gamma = predictions[1]  # Извлекает предсказанное значение gamma
    
    # Проверка, чтобы predicted_gamma не была отрицательной
    if predicted_gamma < 0
        error("Предсказанное значение gamma не может быть отрицательным.")  # Проверка на отрицательное значение
    end
    
    # Проверка, чтобы gamma не выходила за разумные пределы
    gamma = clamp(predicted_gamma, 0.0, 1.0)  # Ограничивает gamma в пределах [0, 1]
    println("Обновляем gamma на основе модели машинного обучения: $gamma")  # Выводит сообщение о новом значении
    return gamma  # Возвращает обновленное значение gamma
end

# Функция для логирования ошибок
function log_error(step::Int, error_message::String, storage::EnergyStorage, rho::Matrix{ComplexF64}, filename::String)
    open(filename, "a") do file
        println(file, "Ошибка на шаге $step:")  # Логирует шаг, на котором произошла ошибка
        println(file, "Сообщение об ошибке: $error_message")  # Логирует сообщение об ошибке
        println(file, "Энергия в накопителе: $(get_current_energy(storage))")  # Логирует энергию накопителя
        println(file, "Плотностная матрица: \n$(string(rho))")  # Логирует плотностную матрицу
        println(file, "======================")  # Разделитель
    end
end

# Основная функция для эксперимента
function quantum_reactor_experiment(dim::Int, gamma::Float64, dt::Float64, steps::Int, num_hamiltonians::Int, start_energy::Float64, final_energy::Float64, storage::EnergyStorage, distributor::EnergyDistributor, run_id::Int, output_dir::String, model::Flux.Chain, coupling::Float64, external_field::Float64)
    println("Инициализация квантового реактора, эксперимент $run_id...")  # Выводит сообщение о начале эксперимента

    entangled_state = create_entangled_state(dim)  # Создает запутанное состояние
    println("Запутанное состояние создано: $entangled_state")  # Выводит информацию о запутанном состоянии

    rho = density_matrix(entangled_state)  # Создает плотностную матрицу
    println("Плотностная матрица: $rho")  # Выводит плотностную матрицу

    Hs = [ising_hamiltonian(dim^2, coupling, external_field) for _ in 1:num_hamiltonians]  # Создает массив Гамильтонианов
    c_ops = create_lindblad_operators(dim^2)  # Создает операторы Линдблада

    successful_run = true  # Флаг успешного завершения эксперимента
    log_filename = joinpath(output_dir, "success", "run_$(run_id)_log.txt")  # Путь к файлу логов успешного выполнения
    error_filename = joinpath(output_dir, "failure", "run_$(run_id)_error.txt")  # Путь к файлу логов ошибок

    anim = @animate for i in 1:steps
        t = i * dt  # Вычисляет текущее время

        if i == 1
            probabilities = abs2.(entangled_state)  # Начальные вероятности
        else
            H = Hs[Int(mod(i-1, num_hamiltonians))+1]  # Выбирает текущий Гамильтониан
            current_energy = time_dependent_energy(start_energy, final_energy, i, steps)  # Вычисляет текущую энергию
            distribute_energy!(distributor, storage, current_energy)  # Распределяет энергию

            try
                rho = evolve_density_operator(rho, H, dt, c_ops, gamma)  # Эволюция плотностной матрицы
            catch e
                println("Ошибка при эволюции плотностной матрицы: $e")  # Логирует ошибку
                log_error(i, string(e), storage, rho, error_filename)  # Логирует ошибку
                successful_run = false  # Обозначает неудачное завершение
                break
            end

            _, probabilities = measure_state(rho)  # Измеряет состояние
            high_prob, mean_prob, var_prob, std_dev_prob = analyze_data(i, probabilities)  # Анализирует данные
            gamma = adjust_parameters!(gamma, high_prob)  # Регулирует gamma

            if high_prob
                log_data(i, get_current_energy(storage), probabilities, rho, log_filename)  # Логирует данные
            end

            # Обновление gamma с использованием модели машинного обучения
            data = rand(Float64, 16)  # Замените на актуальные данные для модели
            gamma = machine_learning_adjustment!(gamma, model, data)  # Регулирует gamma на основе модели
        end

        # Генерация графика на каждом шаге
        frame = plot_state(probabilities, "Шаг $i: вероятность состояний")  # Генерирует график вероятностей
    end

    if successful_run
        mkpath(joinpath(output_dir, "success"))  # Создает директорию для успешных результатов
        gif(anim, joinpath(output_dir, "success", "run_$(run_id).gif"), fps=3)  # Сохраняет анимацию
    else
        mkpath(joinpath(output_dir, "failure"))  # Создает директорию для неудачных результатов
        println("Эксперимент $run_id завершился неудачно.")  # Выводит сообщение о неудачном завершении
    end

    return get_current_energy(storage)  # Возвращает конечную энергию накопителя
end

# Функция для запуска экспериментов
function run_experiments(num_runs::Int, dimension::Int, gamma::Float64, dt::Float64, steps::Int, num_hamiltonians::Int, start_energy::Float64, final_energy::Float64, storage_capacity::Float64, output_dir::String, coupling::Float64, external_field::Float64)
    model = Flux.Chain(Dense(16, 32, relu), Dense(32, 64, relu), Dense(64, 1))  # Создает модель машинного обучения

    for run_id in 1:num_runs
        println("\nЗапуск эксперимента №$run_id...")  # Выводит сообщение о запуске эксперимента

        try
            energy_storage = EnergyStorage(storage_capacity)  # Создает накопитель энергии
            energy_distributor = EnergyDistributor(0.9)  # Создает распределитель энергии

            energy = quantum_reactor_experiment(dimension, gamma, dt, steps, num_hamiltonians, start_energy, final_energy, energy_storage, energy_distributor, run_id, output_dir, model, coupling, external_field)  # Запускает эксперимент
            if energy > 0
                println("Эксперимент завершен для эксперимента №$run_id. Итоговая энергия в накопителе: $(@sprintf("%.3e", energy))")  # Выводит результаты эксперимента
            end
        catch e
            println("Ошибка в эксперименте №$run_id: $e")  # Логирует ошибку
            log_error(-1, string(e), EnergyStorage(storage_capacity), density_matrix(create_superposition_state(dimension)), joinpath(output_dir, "failure", "run_$(run_id)_error.txt"))  # Логирует ошибку
        end
    end
end

# Параметры эксперимента
dimension = 5  # Размерность системы
gamma = 0.4  # Начальное значение gamma
dt = 0.2  # Шаг по времени
steps = 45  # Количество шагов
num_hamiltonians = 5  # Количество Гамильтонианов
start_energy = 100.0  # Начальная энергия
final_energy = 2^50.0  # Конечная энергия
storage_capacity = 2^51.0  # Ёмкость накопителя
num_runs = 100  # Количество экспериментов
output_dir = "output"  # Директория для сохранения результатов
coupling = 0.1  # Параметр взаимодействия
external_field = 0.05  # Внешнее поле

# Создание директорий для сохранения результатов, если их нет
mkpath(joinpath(output_dir, "success"))  # Создает директорию для успешных результатов
mkpath(joinpath(output_dir, "failure"))  # Создает директорию для неудачных результатов

# Запуск экспериментов
run_experiments(num_runs, dimension, gamma, dt, steps, num_hamiltonians, start_energy, final_energy, storage_capacity, output_dir, coupling, external_field)  # Запускает эксперименты

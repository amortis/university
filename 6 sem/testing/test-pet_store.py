import pytest
import requests

BASE_URL = "https://petstore.swagger.io/v2"


# Тесты для работы с Pet
class TestPet:
    def test_add_pet(self):
        """Тестирование добавления нового питомца (POST /pet)"""
        pet_data = {
            "id": 123,
            "name": "Rex",
            "status": "available"
        }
        response = requests.post(f"{BASE_URL}/pet", json=pet_data)

        assert response.status_code == 200
        assert response.json()["id"] == pet_data["id"]
        assert response.json()["name"] == pet_data["name"]
        assert response.json()["status"] == pet_data["status"]

        return pet_data["id"]  # Возвращаем ID для использования в других тестах

    def test_get_pet(self, pet_id):
        """Тестирование получения информации о питомце (GET /pet/{petId})"""
        response = requests.get(f"{BASE_URL}/pet/{pet_id}")

        assert response.status_code == 200
        assert response.json()["id"] == pet_id
        assert "name" in response.json()
        assert "status" in response.json()

    def test_update_pet(self, pet_id):
        """Тестирование обновления питомца (PUT /pet)"""
        updated_data = {
            "id": pet_id,
            "name": "Rex Updated",
            "status": "sold"
        }
        response = requests.put(f"{BASE_URL}/pet", json=updated_data)

        assert response.status_code == 200
        assert response.json()["name"] == "Rex Updated"
        assert response.json()["status"] == "sold"

        # Проверяем через GET, что данные обновились
        get_response = requests.get(f"{BASE_URL}/pet/{pet_id}")
        assert get_response.json()["status"] == "sold"

    def test_delete_pet(self, pet_id):
        """Тестирование удаления питомца (DELETE /pet/{petId})"""
        response = requests.delete(f"{BASE_URL}/pet/{pet_id}")
        assert response.status_code == 200

        # Проверяем, что питомец удален
        get_response = requests.get(f"{BASE_URL}/pet/{pet_id}")
        assert get_response.status_code == 404


# Тесты для работы с Store
class TestStore:
    def test_create_order(self, pet_id):
        """Тестирование создания заказа (POST /store/order)"""
        order_data = {
            "id": 1,
            "petId": pet_id,
            "quantity": 1,
            "status": "placed"
        }
        response = requests.post(f"{BASE_URL}/store/order", json=order_data)

        assert response.status_code == 200
        assert response.json()["petId"] == pet_id
        assert response.json()["status"] == "placed"

        return response.json()["id"]  # Возвращаем ID заказа

    def test_get_order(self, order_id):
        """Тестирование получения заказа (GET /store/order/{orderId})"""
        response = requests.get(f"{BASE_URL}/store/order/{order_id}")

        assert response.status_code == 200
        assert response.json()["id"] == order_id

    def test_delete_order(self, order_id):
        """Тестирование удаления заказа (DELETE /store/order/{orderId})"""
        response = requests.delete(f"{BASE_URL}/store/order/{order_id}")
        assert response.status_code == 200

        # Проверяем, что заказ удален
        get_response = requests.get(f"{BASE_URL}/store/order/{order_id}")
        assert get_response.status_code == 404


# Тесты для работы с User
class TestUser:
    def test_create_user(self):
        """Тестирование создания пользователя (POST /user)"""
        user_data = {
            "id": 1,
            "username": "test_user",
            "email": "user@test.com",
            "password": "test123"
        }
        response = requests.post(f"{BASE_URL}/user", json=user_data)

        assert response.status_code == 200
        return user_data["username"]

    def test_get_user(self, username):
        """Тестирование получения пользователя (GET /user/{username})"""
        response = requests.get(f"{BASE_URL}/user/{username}")

        assert response.status_code == 200
        assert response.json()["username"] == username


# Тесты обработки ошибок
class TestErrorHandling:
    def test_get_nonexistent_pet(self):
        """Тестирование запроса несуществующего питомца"""
        response = requests.get(f"{BASE_URL}/pet/999999")
        assert response.status_code == 404

    def test_create_invalid_pet(self):
        """Тестирование создания питомца с невалидными данными"""
        response = requests.post(f"{BASE_URL}/pet", json={"invalid": "data"})
        assert response.status_code == 500

    def test_delete_nonexistent_order(self):
        """Тестирование удаления несуществующего заказа"""
        response = requests.delete(f"{BASE_URL}/store/order/999999")
        assert response.status_code == 404


# Фикстуры pytest
@pytest.fixture
def pet_id():
    # Создаем питомца для тестов и возвращаем его ID
    pet_data = {
        "id": 123,
        "name": "Rex",
        "status": "available"
    }
    response = requests.post(f"{BASE_URL}/pet", json=pet_data)
    yield pet_data["id"]
    # После завершения теста удаляем питомца
    requests.delete(f"{BASE_URL}/pet/{pet_data['id']}")


@pytest.fixture
def order_id(pet_id):
    # Создаем заказ для тестов и возвращаем его ID
    order_data = {
        "id": 1,
        "petId": pet_id,
        "quantity": 1,
        "status": "placed"
    }
    response = requests.post(f"{BASE_URL}/store/order", json=order_data)
    yield response.json()["id"]
    # После завершения теста удаляем заказ
    requests.delete(f"{BASE_URL}/store/order/{response.json()['id']}")


@pytest.fixture
def username():
    # Создаем пользователя для тестов и возвращаем его username
    user_data = {
        "id": 1,
        "username": "test_user",
        "email": "user@test.com",
        "password": "test123"
    }
    response = requests.post(f"{BASE_URL}/user", json=user_data)
    yield user_data["username"]
    # После завершения теста удаляем пользователя
    requests.delete(f"{BASE_URL}/user/{user_data['username']}")

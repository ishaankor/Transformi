import os
import sys
import asyncio
import random
import graphviz
import numpy as np
import pandas as pd
import pyarrow
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches
import discord
from enum import Enum
from sklearn.linear_model import LinearRegression
from discord.ext import commands
from discord import app_commands
from discord.ui import Select, View, Button, Modal, TextInput
from discord import Intents, File, Embed
from dotenv import load_dotenv
import asyncio
import random
import numpy as np
from discord import File
from discord.ui import Modal, TextInput
from sklearn.linear_model import LinearRegression
from keep_alive import keep_alive
from discord.ui import Modal, TextInput

# Configure SSL for macOS - must be done before any Discord imports
import ssl
import certifi
try:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: ssl_context
    print("SSL certificates configured successfully")
except Exception as e:
    print(f"SSL configuration warning: {e}")

from sklearn.linear_model import LinearRegression


load_dotenv()
bot = commands.Bot(command_prefix='~', intents=Intents.all())
token = os.getenv('DISCORD_BOT_TOKEN')


class BotState:
    def __init__(self):
        self.initialization_status = False
        self.bot_message_history = {}
        self.active_interactions = {}
        # self.createnn_active_interactions = set()
        self.text_channel_list = []

    async def initialize(self):
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)!")

        for guild in bot.guilds:
            for channel in guild.text_channels:
                self.text_channel_list.append(channel)
                try:
                    async for message in channel.history():
                        if message.author == bot.user:
                            self.bot_message_history[message.id] = message
                except discord.errors.Forbidden:
                    pass

        self.initialization_status = True
        print("Ready!")


bot_state = BotState()


@bot.event
async def on_ready():
    await bot_state.initialize()


def initialization_check(ctx: commands.Context) -> bool:
    return bot_state.initialization_status


# @bot.listen()
# async def on_message(message: discord.Message):
#     if message.author != bot.user and message.reference:
#         if message.reference.message_id in bot_state.bot_message_history:
#             await message.reply("Sorry, but this command has already been completed!", delete_after=15)
#             recalled_interaction_message = bot_state.bot_message_history.get(message.reference.message_id)
#             await recalled_interaction_message.delete()
#             await message.delete()
#         else:
#             try:
#                 fetched = await message.channel.fetch_message(message.reference.message_id)
#                 print(f"FETCHED! {fetched.author.bot}")
#                 if fetched.author.bot is True:
#                     # print(message.reference.cached_message)
#                     print("SLEEPING!")
#                     await asyncio.sleep(15)
#                     print("PASSED!")
#                     fetched = await message.channel.fetch_message(message.reference.message_id)
#                     print(fetched)
#                     print(f"{message.author.id} currently in interaction!")
                    # print(message.reference.fail_if_not_exists)

                    # pass ### WORK ON THIS --> GET REFERENCED MESSAGE THEN REVERSE ENGINEER/VERIFY?
                # interaction_ids = [x.id for x in bot_state.active_interactions.values()]
                # if message.reference.message_id not in interaction_ids:
                #     # await asyncio.sleep(10)
                #     try:
                #         await message.channel.fetch_message(message.reference.message_id)
                #     except discord.errors.NotFound:
                #         await message.reply("Thanks for the ping! Cleaning this conversation!", delete_after=5)
                #         await message.delete()
                #         print("Deleted already!")
                # else:
            #         print(f"{message.author.id} currently in interaction!")
            # except:
            #     print("DELETED REFERENCED MESSAGE")
            #     await message.reply("Thanks for the ping! Cleaning this conversation!", delete_after=5)
            #     await message.delete()
            #     print("Deleted!")
            #     pass
                # try:
                #     await message.channel.fetch_message(message.id)
                #     await message.reply("Thanks for the ping! Cleaning this conversation!", delete_after=5)
                #     await message.delete()
                # except:
                #     await message.reply("Thanks for the ping! Cleaning this conversation!", delete_after=5)
                #     await message.delete()
                # print(f"{message.author.id} currently in interaction!")


class RNGModal(Modal, title="Insert the number of data values to create!"):
    def __init__(self):
        super().__init__(timeout=15)

        self.answer = TextInput(
            label="Number of Values",
            style=discord.TextStyle.short,
            required=True, 
            placeholder="Enter a positive integer..."
        )
        self.add_item(self.answer)
        self.response_future = asyncio.get_event_loop().create_future()

    async def on_submit(self, interaction: discord.Interaction):
        """Handles valid user input and generates the graph."""
        try:
            num_values = int(self.answer.value)

            if num_values <= 0:
                raise ValueError("The number must be a positive integer.")

            x_axis = np.arange(1, num_values + 1)
            y_axis = np.random.randint(0, 1000, size=num_values)

            reg = LinearRegression().fit(x_axis.reshape(-1, 1), y_axis)
            plt.scatter(x_axis, y_axis, color="g")
            plt.plot(x_axis, reg.predict(x_axis.reshape(-1, 1)), color="k")
            plt.savefig("test.png")

            test_file = File("test.png")
            await interaction.response.send_message(file=test_file, ephemeral=True)

            self.response_future.set_result(num_values)

        except ValueError:
            await interaction.response.send_message(
                "Invalid input! Please enter a **positive integer**.", ephemeral=True
            )
            self.response_future.set_exception(ValueError("User entered invalid input."))

    async def on_timeout(self):
        """Handles modal timeout (user closes or doesn't respond)."""
        if not self.response_future.done():
            self.response_future.set_exception(asyncio.TimeoutError("Modal timed out."))    


class ManualModal(Modal, title='Insert the array of data values!'):
    def __init__(self):
        super().__init__(timeout=15)
        self.answer = TextInput(label='Array of Values', style=discord.TextStyle.short)
        self.add_item(self.answer)
        self.response_future = asyncio.get_event_loop().create_future()

    async def on_submit(self, interaction: discord.Interaction):
        print("CHECK: " + self.answer.value.replace(' ', ''))
        input_array = [int(x) for x in self.answer.value.replace(' ', '').split(",")]
        print("Transformed array: " + str(input_array))
        try:
            x_axis = list(range(1, len(input_array) + 1))
            np_x = np.array(x_axis)
            np_y = np.array(input_array)

            reg = LinearRegression().fit(np_x.reshape(-1, 1), np_y)

            plt.scatter(np_x, np_y, color='g')
            plt.plot(np_x, reg.predict(np_x.reshape(-1, 1)), color='k')
            plt.savefig('test.png')
            test_file = File('test.png')
            await interaction.response.send_message(file=test_file, ephemeral=True)
            self.response_future.set_result(input_array)
        except AttributeError:
            await interaction.response.send_message(
                "Invalid input! Please in the form of a **comma-separated list of numbers**.", ephemeral=True
            )
            self.response_future.set_exception(ValueError("User entered invalid input."))
            # bot_state.active_interactions.remove(interaction.user.id)

    async def on_timeout(self):
        """Handles modal timeout (user closes or doesn't respond)."""
        if not self.response_future.done():
            self.response_future.set_exception(asyncio.TimeoutError("Modal timed out."))    

class DatasetSelect(Select):
    class ValueIdentification(Enum):
        feature = 1
        label = 2

    def __init__(self, parent_view, dataframe: pd.DataFrame, value_id):
        self.parent_view = parent_view
        self.value_id = value_id
        select_options = [discord.SelectOption(label=column, description=f"This is {column}") for column in dataframe.columns.values]
        super().__init__(placeholder="Choose an option...", min_values=1, max_values=1, options=select_options)

    async def callback(self, interaction: discord.Interaction):
        self.parent_view.feature_input_status = True
        if not self.parent_view.confirmation_status:
            value_id_str = "feature" if self.value_id == DatasetView.ValueIdentification.feature else "label"
            await interaction.response.send_message(
                f"If you've selected the correct {value_id_str}, hit the checkmark above!", ephemeral=True, delete_after=5)
            self.parent_view.current_option = self.values[0]
        else:
            print("All good?!")

class DatasetView(View):
    class ValueIdentification(Enum):
        feature = 1
        label = 2

    def __init__(self, dataframe: pd.DataFrame, feature_or_label: int):
        super().__init__()
        self.confirmation_status = False
        self.feature_input_status = False
        self.current_option = None
        self.selected_option = asyncio.get_event_loop().create_future()
        self.value_id = DatasetView.ValueIdentification(feature_or_label)
        self.add_item(DatasetSelect(self, dataframe, self.value_id))

    @discord.ui.button(emoji="✅")
    async def confirm_feature_callback(self, interaction: discord.Interaction, button: Button):
        if self.feature_input_status:
            self.confirmation_status = True
            for item in self.children:
                item.disabled = True
            await interaction.response.edit_message(view=self)
            self.feature_input_status = False
            self.selected_option.set_result(self.current_option)
            return
        else:
            value_id_str = "feature" if self.value_id == DatasetView.ValueIdentification.feature else "label"
            await interaction.response.send_message(
                f"You did not select any {value_id_str} values! Please select one using the dropdown menu!",
                ephemeral=True, delete_after=5)


async def linear_regression_calculator(interaction, dataframe, feature_set, label_set):
    x_axis = dataframe[feature_set].tolist()
    np_x = np.array(x_axis)
    y_axis = dataframe[label_set].tolist()
    np_y = np.array(y_axis)

    reg = LinearRegression().fit(np_x.reshape(-1, 1), np_y)
    plt.scatter(np_x, np_y, color='g')
    plt.plot(np_x, reg.predict(np_x.reshape(-1, 1)), color='k')
    plt.savefig('test.png')
    test_file = File('test.png')
    await interaction.followup.send(file=test_file, ephemeral=True)


def train_neural_network():
    try:
        
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        x_train = x_train[:5000]
        y_train = y_train[:5000]
        x_test = x_test[:1000]
        y_test = y_test[:1000]
        
        print("Preprocessing data...")
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        print("Creating model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        print("Compiling model...")
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print("Training model...")
        history = model.fit(
            x_train, y_train, 
            epochs=1, 
            validation_data=(x_test, y_test), 
            verbose=0,
            batch_size=32
        )
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        try:
            print("Saving model architecture...")
            plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
            print("Model architecture saved successfully!")
        except Exception as plot_error:
            print(f"Warning: Could not save model plot: {plot_error}")
        
        print("Neural network training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during neural network training: {e}")
        import traceback
        traceback.print_exc()
        return False

async def train(ctx):
    # await ctx.followup.send("Training the neural network, this may take a while...")
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, train_neural_network)
        return result


class GraphLRView(View):
    def __init__(self, original_interaction):
        super().__init__(timeout=30)
        self.original_interaction = original_interaction

    async def on_timeout(self):
        """Prevents timeout from removing an active interaction if the user has already interacted."""
        print(self.original_interaction, bot_state.active_interactions.values())
        if self.original_interaction in bot_state.active_interactions.values():
            print(f"[DEBUG @ 269] Timeout removing active interaction for user {self.original_interaction.user.id}")
            await self.original_interaction.edit_original_response(content="**You didn't respond in time! Try again!**", embed=None, view=None)
            del bot_state.active_interactions[self.original_interaction.user.id]
            return
         
        print(f"[DEBUG @ 274] Timeout ignored for user {self.original_interaction.user.id} since they interacted.")
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
        return


    @discord.ui.button(emoji="1️⃣", label="Random Values", style=discord.ButtonStyle.primary)
    async def button_callback(self, interaction: discord.Interaction, button: Button):
        """Opens the modal and properly handles timeouts and cancellations."""
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        modal = RNGModal()
        await interaction.response.send_modal(modal)
        bot_state.active_interactions[interaction.user.id] = interaction.response

        # for item in self.children:
        #     item.disabled = True
        # await self.original_interaction.edit_original_response(
        #     embed=Embed(title="How would you like to display your dataset?",
        #                 description="1️⃣ Random Dataset Generator \n 2️⃣ Dataset File \n 3️⃣ Given Values/Arrays"), view=self)

        try:
            num_values = await asyncio.wait_for(modal.response_future, timeout=15)  # Wait max 5 min
            print(f"User entered: {num_values}")  # Optional logging
        except asyncio.TimeoutError:
            await interaction.followup.send("You closed the modal or didn’t respond in time.", ephemeral=True)
            print(modal.timeout)
            await self.original_interaction.edit_original_response(view=self)
            print(f"[DEBUG @ 289] Removed active interaction for user {interaction.user.id} due to invalid input.")
        except ValueError:
            pass
        finally:
            del bot_state.active_interactions[interaction.user.id]


    @discord.ui.button(emoji="2️⃣", label="Dataset File (CSV)", style=discord.ButtonStyle.primary)
    async def second_button_callback(self, interaction: discord.Interaction, button: Button):
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        if await interaction_perm_check(interaction):
            await self.original_interaction.edit_original_response(
                embed=Embed(title="How would you like to display your dataset?",
                            description="1️⃣ Random Dataset Generator \n 2️⃣ Dataset File \n 3️⃣ Given Values/Arrays"),
                view=self
            )
            await interaction.response.send_message(
                f"{interaction.user.mention}, please reply to this message with your dataset in a **CSV** file!")
            dataset_prompt = await interaction.original_response()
            print(f"Changed active interaction --> {dataset_prompt}")
            bot_state.active_interactions[interaction.user.id] = dataset_prompt

            async def run(m):
                await m.reply(f"Sorry, this command was not run by you! You can try it by running **/{interaction.command}**!",
                              delete_after=15)
                await m.delete()
                return False

            async def media_reply(message: discord.Message, len_exceeded: bool, file_attached=True):
                if file_attached:
                    if len_exceeded:
                        await dataset_prompt.reply("There's too many files! Please upload just one CSV file!", ephemeral=True)
                        await message.delete()
                        return False
                    else:
                        await dataset_prompt.reply("That's not a valid CSV file! Please upload just one CSV file!", ephemeral=True)
                        await message.delete()
                        return False
                else:
                    await dataset_prompt.reply("There's no files uploaded! Please upload just one CSV file!", ephemeral=True)
                    await message.delete()
                    return False

            def check(m: discord.Message):
                if m.author != bot.user:
                    if m.author == interaction.user:
                        if m.reference and m.reference.message_id == dataset_prompt.id:
                            if len(m.attachments) > 0:
                                if len(m.attachments) == 1 and m.attachments[0].filename.endswith(".csv"):
                                    return True
                                elif len(m.attachments) > 1:
                                    asyncio.create_task(media_reply(m, True))
                                    return False
                                else:
                                    asyncio.create_task(media_reply(m, False))
                                    return False
                            else:
                                asyncio.create_task(media_reply(m, False, False))
                                return False
                        else:
                            print("some other reply?")
                            return False
                    else:
                        if m.reference and m.reference.message_id == dataset_prompt.id:
                            asyncio.create_task(run(m))
                            return False

            try:
                msg = await bot.wait_for("message", check=check, timeout=120.0)
                dataset_file = await msg.attachments[0].to_file()
                print("!")
                await dataset_prompt.delete()
                await msg.delete()
                df = pd.read_csv(dataset_file.fp, engine="pyarrow")
                feature_view = DatasetView(df, 1)
                label_view = DatasetView(df, 2)
                await interaction.followup.send(
                    content="Select the column that represents the feature values!",
                    view=feature_view, ephemeral=True)
                await interaction.followup.send(
                    content="Select the column that represents the label values!",
                    view=label_view, ephemeral=True)
                # await interaction.delete_original_response()
                selected_feature = await feature_view.selected_option
                selected_label = await label_view.selected_option
                print("CHECK!")
                await linear_regression_calculator(interaction, df, selected_feature, selected_label)
            except asyncio.TimeoutError:
                await interaction.followup.send("You took too long to respond! Please run the command again.",
                                                ephemeral=True)
                print("Deleting dataset prompt...")
                await interaction.delete_original_response()
                print(f"[DEBUG @ 380] Removed active interaction for user {interaction.user.id} due to invalid input.")
            finally:
                del bot_state.active_interactions[interaction.user.id]

    
    @discord.ui.button(emoji="3️⃣", label="Manual Input", style=discord.ButtonStyle.primary)
    async def third_button_callback(self, interaction: discord.Interaction, button: Button):
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        modal = ManualModal()
        await interaction.response.send_modal(modal)
        bot_state.active_interactions[interaction.user.id] = interaction.response

        try:
            num_values = await asyncio.wait_for(modal.response_future, timeout=15)  # Wait max 5 min
            print(f"User entered: {num_values}")  # Optional logging
        except asyncio.TimeoutError:
            await interaction.followup.send("You closed the modal or didn’t respond in time.", ephemeral=True)
            print(f"[DEBUG @ 289] Removed active interaction for user {interaction.user.id} due to invalid input.")
        except ValueError:
            pass
        finally:
            del bot_state.active_interactions[interaction.user.id]


class CreateNNView(View):
    def __init__(self, original_interaction):
        super().__init__(timeout=10)
        self.original_interaction = original_interaction
        self.selected_data = None 
    
    async def on_timeout(self):
        """Prevents timeout from removing an active interaction if the user has already interacted."""
        print(self.original_interaction, bot_state.active_interactions.values())
        if self.original_interaction in bot_state.active_interactions.values():
            print(f"[DEBUG @ 426] Timeout removing active interaction for user {self.original_interaction.user.id}")
            await self.original_interaction.edit_original_response(content="**You didn't respond in time! Try again!**", embed=None, view=None)
            del bot_state.active_interactions[self.original_interaction.user.id]
            return
        
        print(f"[DEBUG @ 431] Timeout ignored for user {self.original_interaction.user.id} since they interacted.")
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)
        return


    @discord.ui.button(emoji="1️⃣", label="Random Dataset", style=discord.ButtonStyle.primary)
    async def random_data_callback(self, interaction: discord.Interaction, button: Button):
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        await interaction.response.defer(ephemeral=True)
        bot_state.active_interactions[interaction.user.id] = interaction.response

        num_samples = 1000
        x_axis = np.random.rand(num_samples, 1)
        y_axis = (3 * x_axis + np.random.randn(num_samples, 1) * 0.1).flatten()

        df = pd.DataFrame({"Feature": x_axis.flatten(), "Label": y_axis})
        self.selected_data = df

        await self.start_training(interaction, df)

    @discord.ui.button(emoji="2️⃣", label="Dataset File", style=discord.ButtonStyle.primary)
    async def upload_csv_callback(self, interaction: discord.Interaction, button: Button):
        
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)


        if await interaction_perm_check(interaction):
            await interaction.response.send_message(
                f"{interaction.user.mention}, please reply to this message with your dataset in a **CSV** file!")
            dataset_prompt = await interaction.original_response()
            bot_state.active_interactions[interaction.user.id] = dataset_prompt

            async def run(m):
                await m.reply(f"Sorry, this command was not run by you! You can try it by running **/{interaction.command}**!",
                              delete_after=15)
                await m.delete()
                return False

            async def media_reply(message: discord.Message, len_exceeded: bool, file_attached=True):
                if file_attached:
                    if len_exceeded:
                        await message.reply("There's too many files! Please upload just one CSV file!", delete_after=15)
                        await message.delete()
                        return False
                    else:
                        await message.reply("That's not a valid CSV file! Please upload just one CSV file!",
                                            delete_after=15)
                        await message.delete()
                        return False
                else:
                    await message.reply("There's no files uploaded! Please upload just one CSV file!", delete_after=15)
                    await message.delete()
                    return False

            def check(m: discord.Message):
                if m.author != bot.user:
                    if m.author == interaction.user:
                        if m.reference and m.reference.message_id == dataset_prompt.id:
                            if len(m.attachments) > 0:
                                if len(m.attachments) == 1 and m.attachments[0].filename.endswith(".csv"):
                                    return True
                                elif len(m.attachments) > 1:
                                    asyncio.create_task(media_reply(m, True))
                                    return False
                                else:
                                    asyncio.create_task(media_reply(m, False))
                                    return False
                            else:
                                asyncio.create_task(media_reply(m, False, False))
                                return False
                        else:
                            print("some other reply?")
                            return False
                    else:
                        if m.reference and m.reference.message_id == dataset_prompt.id:
                            asyncio.create_task(run(m))
                            return False

            try:
                msg = await bot.wait_for("message", check=check, timeout=120.0)
                dataset_file = await msg.attachments[0].to_file()
                await dataset_prompt.delete()
                await msg.delete()
                df = pd.read_csv(dataset_file.fp, engine="pyarrow")
                feature_view = DatasetView(df, 1)
                label_view = DatasetView(df, 2)
                await interaction.followup.send(
                    content="Select the column that represents the feature values!",
                    view=feature_view, ephemeral=True)
                await interaction.followup.send(
                    content="Select the column that represents the label values!",
                    view=label_view, ephemeral=True)
                selected_feature = await feature_view.selected_option
                selected_label = await label_view.selected_option
                self.selected_data = df[[selected_feature, selected_label]]
                print("CHECK!")
                await self.start_training(interaction, self.selected_data)
            except asyncio.TimeoutError:
                await interaction.followup.send(
                    "You took too long to upload the file! Please run the command again.",
                    ephemeral=True
                )
                await interaction.delete_original_response()
                print(f"[DEBUG @ 508] Removed active interaction for user {interaction.user.id} due to invalid input.")
                del bot_state.active_interactions[interaction.user.id]

    async def start_training(self, interaction: discord.Interaction, dataframe: pd.DataFrame):
        """Handles the training after dataset selection."""
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        await interaction.followup.send(
            "Training the neural network with your selected dataset. This may take a while...",
            ephemeral=True
        )

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, train_neural_network)  # Run training asynchronously

        if result:
            await interaction.followup.send("Training complete! Model is ready.", ephemeral=True)
        else:
            await interaction.followup.send("Training failed. Please check the logs.", ephemeral=True)
            
        print(f"[DEBUG @ 548] Removed active interaction for user {interaction.user.id} due to invalid input.")
        del bot_state.active_interactions[interaction.user.id]

    @discord.ui.button(emoji="3️⃣", label="Manual Input", style=discord.ButtonStyle.primary)
    async def manual_input_callback(self, interaction: discord.Interaction, button: Button):
        """Let user manually input data values."""
        for item in self.children:
            item.disabled = True
        await self.original_interaction.edit_original_response(view=self)

        modal = ManualModal()
        await interaction.response.send_modal(modal)
        bot_state.active_interactions[interaction.user.id] = interaction.response

async def interaction_perm_check(interaction: discord.Interaction):
    interaction_user_perms = interaction.channel.permissions_for(interaction.user)
    interaction_bot_perms = interaction.channel.permissions_for(interaction.guild.me)
    if interaction_user_perms.attach_files and interaction_bot_perms.attach_files:
        return True
    else:
        await interaction.response.send_message(
            "We can't upload files in this channel if we want to complete the interaction! Please move to a **different channel** or **direct message** me!",
            ephemeral=True)
        print(f"[DEBUG @ 537] Removed active interaction for user {interaction.user.id} due to invalid input.")
        del bot_state.active_interactions[interaction.user.id]
        # bot_state.graphlr_active_interactions.discard(interaction)
        return False

async def check_user_instances(interaction: discord.Interaction):
    """Checks if a user has an ongoing interaction and prevents new ones."""
    user_id = interaction.user.id
    if user_id in bot_state.active_interactions.keys():
        await interaction.response.send_message(
            "You already have an active interaction! Please complete it before starting a new one.",
            ephemeral=True
        )
        return False    
    
    bot_state.active_interactions[user_id] = interaction
    return True


@bot.command(name="restart", description="Restarts the bot if issues are encountered.")
@commands.is_owner()
async def restart(ctx: discord.ext.commands.Context):
    await ctx.reply("Restarting bot...")
    os.execv(sys.executable, ["python"] + sys.argv)


@bot.tree.command(name="graph_linear_regression", description="Graphs a linear regression model of the given dataset/values.")
@app_commands.check(initialization_check)
async def graph_linear_regression(interaction: discord.Interaction):
    if not await check_user_instances(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    graphlr_view = GraphLRView(interaction)
    await interaction.followup.send(embed=Embed(title="How would you like to display your dataset?",
                                                description="1️⃣ Random Dataset Generator \n 2️⃣ Dataset File \n 3️⃣ Given Numbers/Array"),
                                    view=graphlr_view, ephemeral=True)


@bot.tree.command(name="create_neural_network", description="Creates a neural network model of the given dataset/values.")
@app_commands.check(initialization_check)
async def create_neural_network(interaction: discord.Interaction):
    if not await check_user_instances(interaction):
        return

    await interaction.response.defer(ephemeral=True)

    try:
        createnn_view = CreateNNView(interaction)
        await interaction.followup.send(
            embed=Embed(
                title="How would you like to create your neural network?",
                description="1️⃣ Random Dataset \n 2️⃣ Dataset File \n 3️⃣ Manual Input"
            ),
            view=createnn_view,
            ephemeral=True
        )
    except Exception as e:
        print(f"Error in create_neural_network: {e}")
        await interaction.followup.send("An error occurred while processing your request.", ephemeral=True)



@bot.event
async def on_command_error(ctx, exception):
    if isinstance(exception, commands.CommandOnCooldown):
        await ctx.reply(f"You are rate limited. Please, try again in {exception.retry_after} seconds")
    elif isinstance(exception, commands.CheckFailure):
        await ctx.reply("You don't have the necessary permissions to run this command!")
    else:
        print(f"Unhandled exception: {exception}")


@graph_linear_regression.error
async def graph_linear_regression_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        await interaction.response.send_message("Initialization is not complete. Please try again later.",
                                                ephemeral=True)
    else:
        print(f"[DEBUG @ 617] Removed active interaction for user {interaction.user.id} due to invalid input.")
        del bot_state.active_interactions[interaction.user.id]
        await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)


@create_neural_network.error
async def create_neural_network_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        await interaction.response.send_message("Initialization is not complete. Please try again later.",
                                                ephemeral=True)
    else:
        print(f"[DEBUG @ 628] Removed active interaction for user {interaction.user.id} due to invalid input.")
        del bot_state.active_interactions[interaction.user.id]
        await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)


keep_alive()
bot.run(token)
